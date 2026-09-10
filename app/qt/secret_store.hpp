#ifndef SIRIUS_APP_SECRET_STORE_HPP
#define SIRIUS_APP_SECRET_STORE_HPP

// Where the application keeps the secrets the user types into it: the HPC
// worker token, the Hugging Face token and the assistant's API key.
//
// They used to sit in QSettings as plain text -- the Windows registry under
// HKCU\Software\..., a world-readable ~/.config/*.conf on Linux -- which put
// them in every settings backup and in front of anything that reads the user's
// configuration. The store keeps the same key names ("hpc/token",
// "hub/token", "assistant/apiKey") so a call site only changes which function
// it calls, and read() migrates a plaintext value it still finds there:
// it moves the value into the store and deletes the old entry.
//
// Windows: DPAPI (CryptProtectData) with the key name as entropy, base64 in
// QSettings under secrets/<key>. Only this user on this machine can read it
// back, and only under the key it was written for.
//
// Everywhere else: ~/.sirius/secrets.json, created 0600. The values in that
// file are obfuscated, NOT encrypted -- the obfuscation only keeps the token
// out of a `grep -r` and out of a backup that someone skims. The file mode is
// the actual protection; anyone who can read the file can recover the value.
// A keyring (libsecret / Keychain) would be the real fix and needs a
// dependency the project does not have yet.

#include <QString>

namespace sirius::app::secrets {

    // The stored secret, or an empty string when there is none. Migrates a
    // plaintext QSettings value under the same key on the way.
    QString read(const QString& key);

    // Stores `value`, or removes the secret when `value` is empty. Also
    // clears any plaintext leftover under the same QSettings key. False when
    // the backend refused (DPAPI, or the store file could not be written):
    // the caller then knows the value is gone at the next launch.
    bool write(const QString& key, const QString& value);

    void remove(const QString& key);

} // namespace sirius::app::secrets

#endif // SIRIUS_APP_SECRET_STORE_HPP
