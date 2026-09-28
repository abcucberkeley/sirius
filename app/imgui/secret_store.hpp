#ifndef SIRIUS_IMGUI_SECRET_STORE_HPP
#define SIRIUS_IMGUI_SECRET_STORE_HPP

// Where the application keeps the secrets the user types into it: the HPC
// worker token, the Hugging Face token and the assistant's API key. Never
// the settings file as plain text. The key names are the Qt application's
// ("hpc/token", "hub/token", "assistant/apiKey").
//
// Windows: DPAPI (CryptProtectData) with the key name as entropy, base64 in
// the settings under secrets/<key>. Only this user on this machine can read
// it back, and only under the key it was written for.
//
// Everywhere else: ~/.sirius/secrets.json, created 0600 -- the file the Qt
// application uses, in its format, so a token entered in one is there in the
// other. The values in that file are obfuscated, NOT encrypted: the file
// mode is the actual protection. The file is replaced atomically, and one
// that exists but does not parse is never written over.

#include <string>

namespace sirius::app::gui::secrets {

    // The stored secret, or an empty string when there is none. Migrates a
    // plaintext settings value under the same key on the way.
    std::string read(const std::string& key);

    // Stores `value`, or removes the secret when `value` is empty. False when
    // the backend refused: the caller then knows the value is gone at the
    // next launch.
    bool write(const std::string& key, const std::string& value);

    // False when the store could not be rewritten without the secret.
    bool remove(const std::string& key);

    // Keeps the file store in `dir` instead of ~/.sirius; an empty `dir` goes
    // back to ~/.sirius. main() calls it with --settings, before the first read.
    void setStoreDirectory(const std::string& dir);

    // Base64 (RFC 4648, padded), shared with whoever needs it.
    std::string toBase64(const std::string& bytes);
    std::string fromBase64(const std::string& text);

} // namespace sirius::app::gui::secrets

#endif // SIRIUS_IMGUI_SECRET_STORE_HPP
