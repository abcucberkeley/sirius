#include "qt/secret_store.hpp"

#include <QByteArray>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>
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

        void removeBackend(const QString& key) { QSettings().remove(settingsKey(key)); }

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

        QString storePath() { return QDir::homePath() + QStringLiteral("/.sirius/secrets.json"); }

        QJsonObject loadStore() {
            QFile f(storePath());
            if (!f.open(QIODevice::ReadOnly)) return QJsonObject();
            const QJsonDocument doc = QJsonDocument::fromJson(f.readAll());
            return doc.isObject() ? doc.object() : QJsonObject();
        }

        bool saveStore(const QJsonObject& obj) {
            const QString path = storePath();
            QDir().mkpath(QFileInfo(path).absolutePath());
            QFile::setPermissions(QFileInfo(path).absolutePath(),
                                  QFileDevice::ReadOwner | QFileDevice::WriteOwner | QFileDevice::ExeOwner);
            QFile f(path);
            if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) return false;
            // Tighten the mode on the (still empty) file before anything is
            // written into it, so the secret is never briefly world-readable.
            f.setPermissions(QFileDevice::ReadOwner | QFileDevice::WriteOwner);
            const QByteArray text = QJsonDocument(obj).toJson(QJsonDocument::Indented);
            const bool written = f.write(text) == text.size();
            f.close();
            return written && f.error() == QFileDevice::NoError;
        }

        QString readBackend(const QString& key) {
            const QJsonValue v = loadStore().value(key);
            if (!v.isString()) return QString();
            return QString::fromUtf8(mask(QByteArray::fromBase64(v.toString().toLatin1()), key));
        }

        bool writeBackend(const QString& key, const QString& value) {
            QJsonObject obj = loadStore();
            obj.insert(key, QString::fromLatin1(mask(value.toUtf8(), key).toBase64()));
            return saveStore(obj);
        }

        void removeBackend(const QString& key) {
            QJsonObject obj = loadStore();
            if (obj.contains(key)) {
                obj.remove(key);
                saveStore(obj);
            }
        }

#endif

    } // namespace

    QString read(const QString& key) {
        const QString stored = readBackend(key);
        if (!stored.isEmpty()) return stored;

        // Migration from the plaintext QSettings entry this store replaced.
        QSettings s;
        const QString legacy = s.value(key).toString();
        if (legacy.isEmpty()) return QString();
        writeBackend(key, legacy);
        s.remove(key);
        return legacy;
    }

    bool write(const QString& key, const QString& value) {
        bool ok = true;
        if (value.isEmpty())
            removeBackend(key);
        else
            ok = writeBackend(key, value);
        QSettings().remove(key);   // never leave the old plaintext behind
        if (!ok) qWarning("secret store: could not store '%s'", qPrintable(key));
        return ok;
    }

    void remove(const QString& key) {
        removeBackend(key);
        QSettings().remove(key);
    }

} // namespace sirius::app::secrets
