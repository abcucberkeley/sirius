#include "qt/secret_store.hpp"

#include <mutex>

#include <QByteArray>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QJsonValue>
#include <QSaveFile>
#include <QSet>
#include <QSettings>

#ifdef Q_OS_WIN
// After the Qt headers, and with the macros Windows would otherwise inject.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
// WIN32_LEAN_AND_MEAN drops the crypto headers, so they come in by hand;
// dpapi.h (CryptProtectData) arrives with wincrypt.h.
#include <wincrypt.h>
#endif

namespace sirius::app::secrets {

    namespace {

        // Where the file store lives when a run keeps settings of its own
        // (setStoreDirectory); empty for ~/.sirius.
        QString& storeDirectory() {
            static QString dir;
            return dir;
        }

#ifdef Q_OS_WIN

        // The QSettings subtree the DPAPI blobs live in, next to (but not on
        // top of) the plaintext key the migration reads. Only this backend
        // keeps anything in QSettings; the file backend below has no use for it.
        QString settingsKey(const QString& key) { return QStringLiteral("secrets/") + key; }

        DATA_BLOB blobOf(QByteArray& bytes) {
            DATA_BLOB b;
            b.cbData = static_cast<DWORD>(bytes.size());
            b.pbData = reinterpret_cast<BYTE*>(bytes.data());
            return b;
        }

        // DPAPI, tied to this user on this machine. The key name goes in as
        // entropy, so a blob copied from one setting to another will not
        // decrypt. CRYPTPROTECT_UI_FORBIDDEN: never block on a prompt.
        QByteArray protect(const QByteArray& plain, const QString& key) {
            QByteArray in = plain;
            QByteArray entropy = key.toUtf8();
            DATA_BLOB inBlob = blobOf(in);
            DATA_BLOB entropyBlob = blobOf(entropy);
            DATA_BLOB out{};
            if (!::CryptProtectData(&inBlob, L"SIRIUS", &entropyBlob, nullptr, nullptr, CRYPTPROTECT_UI_FORBIDDEN, &out))
                return QByteArray();
            const QByteArray result(reinterpret_cast<const char*>(out.pbData), static_cast<int>(out.cbData));
            ::LocalFree(out.pbData);
            return result;
        }

        QByteArray unprotect(const QByteArray& blob, const QString& key) {
            QByteArray in = blob;
            QByteArray entropy = key.toUtf8();
            DATA_BLOB inBlob = blobOf(in);
            DATA_BLOB entropyBlob = blobOf(entropy);
            DATA_BLOB out{};
            if (!::CryptUnprotectData(&inBlob, nullptr, &entropyBlob, nullptr, nullptr, CRYPTPROTECT_UI_FORBIDDEN, &out))
                return QByteArray();
            const QByteArray result(reinterpret_cast<const char*>(out.pbData), static_cast<int>(out.cbData));
            ::LocalFree(out.pbData);
            return result;
        }

        QString readBackend(const QString& key) {
            QSettings s;
            const QByteArray blob = QByteArray::fromBase64(s.value(settingsKey(key)).toString().toLatin1());
            if (blob.isEmpty()) return QString();
            return QString::fromUtf8(unprotect(blob, key));
        }

        bool writeBackend(const QString& key, const QString& value) {
            QSettings s;
            const QByteArray blob = protect(value.toUtf8(), key);
            if (blob.isEmpty()) return false;   // DPAPI refused; better no value than a plaintext one
            s.setValue(settingsKey(key), QString::fromLatin1(blob.toBase64()));
            return true;
        }

        bool removeBackend(const QString& key) {
            QSettings().remove(settingsKey(key));
            return true;
        }

#else

        // NOT encryption: see the header. The mask only keeps the token from
        // being readable in a file someone opens or greps by accident.
        QByteArray mask(const QByteArray& in, const QString& key) {
            const QByteArray salt = QByteArrayLiteral("sirius/secrets/v1/") + key.toUtf8();
            QByteArray out = in;
            for (int i = 0; i < out.size(); ++i)
                out[i] = static_cast<char>(out.at(i) ^ salt.at(i % salt.size()) ^ static_cast<char>(i & 0xff));
            return out;
        }

        QString storePath() {
            const QString dir = storeDirectory().isEmpty() ? QDir::homePath() + QStringLiteral("/.sirius") : storeDirectory();
            return dir + QStringLiteral("/secrets.json");
        }

        // The store as it is on disk: an empty object when there is no file
        // (or an empty one). False when the file is there but cannot be read
        // or parsed -- a write would then replace every secret in it with
        // the one being written, so the callers refuse instead.
        bool loadStore(QJsonObject& obj) {
            obj = QJsonObject();
            QFile f(storePath());
            if (!f.exists()) return true;
            if (!f.open(QIODevice::ReadOnly)) return false;
            const QByteArray text = f.readAll();
            if (text.trimmed().isEmpty()) return true;
            QJsonParseError error;
            const QJsonDocument doc = QJsonDocument::fromJson(text, &error);
            if (error.error != QJsonParseError::NoError || !doc.isObject()) return false;
            obj = doc.object();
            return true;
        }

        bool saveStore(const QJsonObject& obj) {
            const QString path = storePath();
            QDir().mkpath(QFileInfo(path).absolutePath());
            QFile::setPermissions(QFileInfo(path).absolutePath(),
                                  QFileDevice::ReadOwner | QFileDevice::WriteOwner | QFileDevice::ExeOwner);
            // Written to a file beside the store and renamed over it by
            // commit(): a short write or a crash leaves the previous store
            // whole, where truncating it in place left an empty file and
            // every secret gone.
            QSaveFile f(path);
            if (!f.open(QIODevice::WriteOnly)) return false;
            // Tighten the mode on the (still empty) file before anything is
            // written into it, so the secret is never briefly world-readable.
            f.setPermissions(QFileDevice::ReadOwner | QFileDevice::WriteOwner);
            const QByteArray text = QJsonDocument(obj).toJson(QJsonDocument::Indented);
            if (f.write(text) != text.size()) {
                f.cancelWriting();
                return false;
            }
            return f.commit();
        }

        QString readBackend(const QString& key) {
            QJsonObject obj;
            loadStore(obj);   // an unreadable store holds nothing this can use
            const QJsonValue v = obj.value(key);
            if (!v.isString()) return QString();
            return QString::fromUtf8(mask(QByteArray::fromBase64(v.toString().toLatin1()), key));
        }

        bool writeBackend(const QString& key, const QString& value) {
            QJsonObject obj;
            if (!loadStore(obj)) return false;
            obj.insert(key, QString::fromLatin1(mask(value.toUtf8(), key).toBase64()));
            return saveStore(obj);
        }

        bool removeBackend(const QString& key) {
            QJsonObject obj;
            if (!loadStore(obj)) return false;
            if (!obj.contains(key)) return true;
            obj.remove(key);
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
        // said once per key, not on every read (the hub token is read for
        // each request).
        void warnKeptPlaintext(const QString& key) {
            static QSet<QString> warned;
            if (warned.contains(key)) return;
            warned.insert(key);
            qWarning("secret store: could not move '%s' into the store; it stays in the settings as plain text until it can be",
                     qPrintable(key));
        }

    } // namespace

    QString read(const QString& key) {
        const std::lock_guard<std::mutex> lock(storeMutex());
        const QString stored = readBackend(key);
        if (!stored.isEmpty()) return stored;

        // Migration from the plaintext QSettings entry this store replaced.
        // The entry goes only once the store holds the value: deleting it
        // after a refused write (DPAPI, a read-only or unparsable store)
        // worked for this session and lost the token at the next launch.
        QSettings s;
        const QString legacy = s.value(key).toString();
        if (legacy.isEmpty()) return QString();
        if (writeBackend(key, legacy)) s.remove(key);
        else warnKeptPlaintext(key);
        return legacy;
    }

    bool write(const QString& key, const QString& value) {
        const std::lock_guard<std::mutex> lock(storeMutex());
        const bool ok = value.isEmpty() ? removeBackend(key) : writeBackend(key, value);
        // Never leave an old plaintext value behind -- unless the store
        // refused this very value and the plaintext is its only copy.
        QSettings s;
        if (ok || s.value(key).toString() != value) s.remove(key);
        if (!ok) qWarning("secret store: could not store '%s'", qPrintable(key));
        return ok;
    }

    bool remove(const QString& key) {
        const std::lock_guard<std::mutex> lock(storeMutex());
        const bool ok = removeBackend(key);
        QSettings().remove(key);
        return ok;
    }

    void setStoreDirectory(const QString& dir) {
        const std::lock_guard<std::mutex> lock(storeMutex());
        storeDirectory() = dir;
    }

} // namespace sirius::app::secrets
