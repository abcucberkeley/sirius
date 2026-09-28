#include "imgui/secret_store.hpp"

#include <cstdio>
#include <mutex>
#include <set>

#include <nlohmann/json.hpp>

#ifdef _WIN32
#include <windows.h>
// WIN32_LEAN_AND_MEAN drops the crypto headers, so they come in by hand;
// dpapi.h (CryptProtectData) arrives with wincrypt.h.
#include <wincrypt.h>
#else
#include <sys/stat.h>
#endif

#include "imgui/platform.hpp"
#include "imgui/settings.hpp"
#include "imgui/strings.hpp"

namespace sirius::app::gui::secrets {

    std::string toBase64(const std::string& bytes) {
        static const char* table = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::string out;
        out.reserve((bytes.size() + 2) / 3 * 4);
        std::size_t i = 0;
        while (i + 2 < bytes.size()) {
            const unsigned v = (static_cast<unsigned char>(bytes[i]) << 16) | (static_cast<unsigned char>(bytes[i + 1]) << 8) |
                               static_cast<unsigned char>(bytes[i + 2]);
            out += table[(v >> 18) & 63];
            out += table[(v >> 12) & 63];
            out += table[(v >> 6) & 63];
            out += table[v & 63];
            i += 3;
        }
        if (i + 1 == bytes.size()) {
            const unsigned v = static_cast<unsigned char>(bytes[i]) << 16;
            out += table[(v >> 18) & 63];
            out += table[(v >> 12) & 63];
            out += "==";
        } else if (i + 2 == bytes.size()) {
            const unsigned v = (static_cast<unsigned char>(bytes[i]) << 16) | (static_cast<unsigned char>(bytes[i + 1]) << 8);
            out += table[(v >> 18) & 63];
            out += table[(v >> 12) & 63];
            out += table[(v >> 6) & 63];
            out += '=';
        }
        return out;
    }

    std::string fromBase64(const std::string& text) {
        std::string out;
        unsigned acc = 0;
        int bits = 0;
        for (char ch : text) {
            int v = -1;
            if (ch >= 'A' && ch <= 'Z') v = ch - 'A';
            else if (ch >= 'a' && ch <= 'z') v = ch - 'a' + 26;
            else if (ch >= '0' && ch <= '9') v = ch - '0' + 52;
            else if (ch == '+' || ch == '-') v = 62;
            else if (ch == '/' || ch == '_') v = 63;
            else continue;   // padding, whitespace
            acc = (acc << 6) | static_cast<unsigned>(v);
            bits += 6;
            if (bits >= 8) {
                bits -= 8;
                out += static_cast<char>((acc >> bits) & 0xFF);
            }
        }
        return out;
    }

    namespace {

        // Where the file store lives when a run keeps settings of its own
        // (setStoreDirectory); empty for ~/.sirius.
        std::string& storeDirectory() {
            static std::string dir;
            return dir;
        }

#ifdef _WIN32

        // The settings subtree the DPAPI blobs live in, next to (but not on
        // top of) the plaintext key the migration reads.
        std::string settingsKey(const std::string& key) { return "secrets/" + key; }

        DATA_BLOB blobOf(std::string& bytes) {
            DATA_BLOB b;
            b.cbData = static_cast<DWORD>(bytes.size());
            b.pbData = reinterpret_cast<BYTE*>(bytes.data());
            return b;
        }

        // DPAPI, tied to this user on this machine. The key name goes in as
        // entropy, so a blob copied from one setting to another will not
        // decrypt. CRYPTPROTECT_UI_FORBIDDEN: never block on a prompt.
        std::string protect(const std::string& plain, const std::string& key) {
            std::string in = plain, entropy = key;
            DATA_BLOB inBlob = blobOf(in), entropyBlob = blobOf(entropy), out{};
            if (!::CryptProtectData(&inBlob, L"SIRIUS", &entropyBlob, nullptr, nullptr, CRYPTPROTECT_UI_FORBIDDEN, &out))
                return std::string();
            const std::string result(reinterpret_cast<const char*>(out.pbData), static_cast<std::size_t>(out.cbData));
            ::LocalFree(out.pbData);
            return result;
        }

        std::string unprotect(const std::string& blob, const std::string& key) {
            std::string in = blob, entropy = key;
            DATA_BLOB inBlob = blobOf(in), entropyBlob = blobOf(entropy), out{};
            if (!::CryptUnprotectData(&inBlob, nullptr, &entropyBlob, nullptr, nullptr, CRYPTPROTECT_UI_FORBIDDEN, &out))
                return std::string();
            const std::string result(reinterpret_cast<const char*>(out.pbData), static_cast<std::size_t>(out.cbData));
            ::LocalFree(out.pbData);
            return result;
        }

        std::string readBackend(const std::string& key) {
            const std::string blob = fromBase64(settings().getString(settingsKey(key)));
            if (blob.empty()) return std::string();
            return unprotect(blob, key);
        }

        bool writeBackend(const std::string& key, const std::string& value) {
            const std::string blob = protect(value, key);
            if (blob.empty()) return false;   // DPAPI refused; better no value than a plaintext one
            settings().set(settingsKey(key), toBase64(blob));
            settings().save();
            return true;
        }

        bool removeBackend(const std::string& key) {
            settings().remove(settingsKey(key));
            settings().save();
            return true;
        }

#else

        // NOT encryption: see the header. The mask only keeps the token from
        // being readable in a file someone opens or greps by accident.
        std::string mask(const std::string& in, const std::string& key) {
            const std::string salt = "sirius/secrets/v1/" + key;
            std::string out = in;
            for (std::size_t i = 0; i < out.size(); ++i)
                out[i] = static_cast<char>(out[i] ^ salt[i % salt.size()] ^ static_cast<char>(i & 0xff));
            return out;
        }

        std::string storeDir() { return storeDirectory().empty() ? platform::homeDirectory() + "/.sirius" : storeDirectory(); }
        std::string storePath() { return storeDir() + "/secrets.json"; }

        // The store as it is on disk: an empty object when there is no file
        // (or an empty one). False when the file is there but cannot be read
        // or parsed -- a write would then replace every secret in it with
        // the one being written, so the callers refuse instead.
        bool loadStore(nlohmann::json& obj) {
            obj = nlohmann::json::object();
            if (!pathExists(storePath())) return true;
            std::string text;
            if (!platform::readFile(storePath(), text)) return false;
            if (trimmed(text).empty()) return true;
            const nlohmann::json j = nlohmann::json::parse(text, nullptr, false);
            if (!j.is_object()) return false;
            obj = j;
            return true;
        }

        bool saveStore(const nlohmann::json& obj) {
            platform::makePath(storeDir());
            ::chmod(storeDir().c_str(), S_IRWXU);
            return platform::writeFileAtomic(storePath(), obj.dump(4) + "\n", true);
        }

        std::string readBackend(const std::string& key) {
            nlohmann::json obj;
            loadStore(obj);   // an unreadable store holds nothing this can use
            const auto it = obj.find(key);
            if (it == obj.end() || !it->is_string()) return std::string();
            return mask(fromBase64(it->get<std::string>()), key);
        }

        bool writeBackend(const std::string& key, const std::string& value) {
            nlohmann::json obj;
            if (!loadStore(obj)) return false;
            obj[key] = toBase64(mask(value, key));
            return saveStore(obj);
        }

        bool removeBackend(const std::string& key) {
            nlohmann::json obj;
            if (!loadStore(obj)) return false;
            if (!obj.contains(key)) return true;
            obj.erase(key);
            return saveStore(obj);
        }

#endif

        // One store for the process: the hub token is read on a run's thread
        // while the GUI may be writing, and the file backend's
        // read-modify-write must not interleave.
        std::mutex& storeMutex() {
            static std::mutex m;
            return m;
        }

        // A plaintext value the store would not take is left where it is and
        // said once per key, not on every read.
        void warnKeptPlaintext(const std::string& key) {
            static std::set<std::string> warned;
            if (!warned.insert(key).second) return;
            std::fprintf(stderr,
                         "secret store: could not move '%s' into the store; it stays in the settings as plain text until it can be\n",
                         key.c_str());
        }

    } // namespace

    std::string read(const std::string& key) {
        const std::lock_guard<std::mutex> lock(storeMutex());
        const std::string stored = readBackend(key);
        if (!stored.empty()) return stored;

        // Migration from a plaintext settings entry. The entry goes only once
        // the store holds the value.
        const std::string legacy = settings().getString(key);
        if (legacy.empty()) return std::string();
        if (writeBackend(key, legacy)) {
            settings().remove(key);
            settings().save();
        } else {
            warnKeptPlaintext(key);
        }
        return legacy;
    }

    bool write(const std::string& key, const std::string& value) {
        const std::lock_guard<std::mutex> lock(storeMutex());
        const bool ok = value.empty() ? removeBackend(key) : writeBackend(key, value);
        // Never leave an old plaintext value behind -- unless the store
        // refused this very value and the plaintext is its only copy.
        if (ok || settings().getString(key) != value) {
            settings().remove(key);
            settings().save();
        }
        if (!ok) std::fprintf(stderr, "secret store: could not store '%s'\n", key.c_str());
        return ok;
    }

    bool remove(const std::string& key) {
        const std::lock_guard<std::mutex> lock(storeMutex());
        const bool ok = removeBackend(key);
        settings().remove(key);
        settings().save();
        return ok;
    }

    void setStoreDirectory(const std::string& dir) {
        const std::lock_guard<std::mutex> lock(storeMutex());
        storeDirectory() = dir;
    }

} // namespace sirius::app::gui::secrets
