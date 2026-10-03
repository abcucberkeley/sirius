// A folder of TIFF stacks on the cluster as one dataset: listed over the
// session's command channel, described by the same manifest as a folder on
// this computer, with the stacks' shapes asked of the engine on the node, and
// the manifest kept in ~/.sirius/manifests there -- never in the data folder.
#include "core/cluster_folder.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

#include "core/array_source.hpp"
#include "core/remote_host.hpp"
#include "core/remote_source.hpp"

namespace sirius::app {

    using json = nlohmann::json;

    namespace {

        // A stand-in cluster on Windows answers "C:\x\y": the same path with
        // forward slashes, as cluster:// paths are written.
        std::string clusterSlashes(std::string p) {
            if (p.size() > 1 && p[1] == ':')
                for (char& c : p)
                    if (c == '\\') c = '/';
            return p;
        }

        std::string joined(const std::string& folder, const std::string& name) {
            if (folder.empty()) return name;
            return folder.back() == '/' ? folder + name : folder + "/" + name;
        }

        std::string lastComponent(std::string p) {
            while (p.size() > 1 && p.back() == '/') p.pop_back();
            const std::size_t slash = p.find_last_of('/');
            return slash == std::string::npos ? p : p.substr(slash + 1);
        }

        std::string base64(const std::string& in) {
            static const char* k = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            std::string out;
            out.reserve((in.size() + 2) / 3 * 4);
            std::size_t i = 0;
            for (; i + 2 < in.size(); i += 3) {
                const std::uint32_t v = (static_cast<std::uint32_t>(static_cast<unsigned char>(in[i])) << 16) |
                                        (static_cast<std::uint32_t>(static_cast<unsigned char>(in[i + 1])) << 8) |
                                        static_cast<std::uint32_t>(static_cast<unsigned char>(in[i + 2]));
                out += k[(v >> 18) & 63];
                out += k[(v >> 12) & 63];
                out += k[(v >> 6) & 63];
                out += k[v & 63];
            }
            if (i < in.size()) {
                std::uint32_t v = static_cast<std::uint32_t>(static_cast<unsigned char>(in[i])) << 16;
                if (i + 1 < in.size()) v |= static_cast<std::uint32_t>(static_cast<unsigned char>(in[i + 1])) << 8;
                out += k[(v >> 18) & 63];
                out += k[(v >> 12) & 63];
                out += i + 1 < in.size() ? k[(v >> 6) & 63] : '=';
                out += '=';
            }
            return out;
        }

        // The last line of `output` that is a JSON object.
        json lastJsonLine(const std::string& output, const char* what) {
            std::istringstream in(output);
            std::string line, last;
            while (std::getline(in, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (!line.empty() && line.front() == '{') last = line;
            }
            try {
                return json::parse(last);
            } catch (const json::exception&) {
                throw ssh::SshError(std::string("the cluster's answer to ") + what + " was not readable", output.substr(0, 400));
            }
        }

        const char* kPythonPrelude = "P=$(command -v python3 || command -v python) || { echo '{\"error\": \"no python3 on this host\"}'; exit 0; }\n";

    } // namespace

    std::string ClusterFolder::clusterPath() const { return makeClusterPath(host, path); }

    std::string ClusterFolder::clusterPathOf(const std::string& name) const { return makeClusterPath(host, joined(path, name)); }

    std::string clusterFolderScript(const std::string& path, int cap) {
        const std::string word = path.empty() ? std::string("\"$HOME\"") : ssh::remotePathWord(path);
        return std::string(kPythonPrelude) + "\"$P\" - " + word + " " + std::to_string(cap) + " \"$HOME\"" +
               " <<'SIRIUS_FOLDER'\n"
               "import json, os, sys\n"
               "def clean(s):\n"
               "    try:\n"
               "        s.encode('utf-8')\n"
               "        return s\n"
               "    except UnicodeEncodeError:\n"
               "        return s.encode('utf-8', 'surrogateescape').decode('utf-8', 'replace')\n"
               "p = os.path.abspath(sys.argv[1])\n"
               "cap = int(sys.argv[2])\n"
               "home = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else os.path.expanduser('~')\n"
               "out = {'path': clean(p), 'home': clean(home), 'tiffs': [], 'others': 0, 'truncated': False, "
               "'store': False}\n"
               "try:\n"
               "    with os.scandir(p) as it:\n"
               "        for e in it:\n"
               "            n = e.name\n"
               "            if n in ('.zarray', '.zgroup', 'zarr.json', 'attributes.json'):\n"
               "                out['store'] = True\n"
               "            if n == 'sirius-dataset.toml':\n"
               "                try:\n"
               "                    with open(os.path.join(p, n), 'rb') as f:\n"
               "                        out['manifest'] = clean(f.read(8 << 20).decode('utf-8', 'surrogateescape'))\n"
               "                except OSError as x:\n"
               "                    out['manifest_error'] = '%s: %s' % (n, x.strerror or x)\n"
               "                continue\n"
               "            if n.startswith('.'):\n"
               "                continue\n"
               "            low = n.lower()\n"
               "            if not (low.endswith('.tif') or low.endswith('.tiff')):\n"
               "                out['others'] += 1\n"
               "                continue\n"
               "            try:\n"
               "                if e.is_dir():\n"
               "                    continue\n"
               "            except OSError:\n"
               "                pass\n"
               "            if len(out['tiffs']) >= cap:\n"
               "                out['truncated'] = True\n"
               "                continue\n"
               "            out['tiffs'].append(clean(n))\n"
               "except OSError as x:\n"
               "    out = {'error': '%s: %s' % (clean(p), x.strerror or x)}\n"
               "sys.stdout.write(json.dumps(out) + '\\n')\n"
               "SIRIUS_FOLDER\n";
    }

    ClusterFolder parseClusterFolder(const std::string& output) {
        const json j = lastJsonLine(output, "a folder listing");
        if (j.contains("error")) throw ssh::SshError(j["error"].is_string() ? j["error"].get<std::string>() : "cannot list the folder");
        ClusterFolder f;
        f.path = clusterSlashes(j.value("path", std::string()));
        f.home = clusterSlashes(j.value("home", std::string()));
        f.others = j.value("others", std::size_t{0});
        f.truncated = j.value("truncated", false);
        f.store = j.value("store", false);
        std::vector<std::string> names;
        if (j.contains("tiffs") && j["tiffs"].is_array())
            for (const json& n : j["tiffs"])
                if (n.is_string()) names.push_back(n.get<std::string>());
        f.tiffs = tiffNamesOf(std::move(names));
        if (j.contains("manifest") && j["manifest"].is_string()) {
            try {
                f.existing = DatasetManifest::fromText(j["manifest"].get<std::string>(), joined(f.path, DatasetManifest::kFileName));
            } catch (const std::exception& e) {
                f.existingError = e.what();
            }
        } else if (j.contains("manifest_error") && j["manifest_error"].is_string()) {
            f.existingError = j["manifest_error"].get<std::string>();
        }
        return f;
    }

    ClusterFolder listClusterFolder(cluster::Session& session, const std::string& host, const std::string& path, int cap) {
        const ssh::CommandResult r = session.run(clusterFolderScript(path, cap), std::chrono::seconds(180));
        ClusterFolder f = parseClusterFolder(r.out);
        f.host = host;
        return f;
    }

    DatasetManifest manifestFromClusterFolder(const ClusterFolder& folder, const FilenameRule& rule, std::vector<std::string>* unmatched) {
        // the engine on the node reads each file's header: dims and pages
        // as it opens the file, which is how the manifest's files are read
        const StackShapeProbe probe = [&folder](const std::string& name) {
            const DatasetMeta m = probeDataset(folder.clusterPathOf(name));
            StackShape s;
            s.width = static_cast<std::uint32_t>(m.dims.x);
            s.height = static_cast<std::uint32_t>(m.dims.y);
            s.pages = static_cast<std::size_t>(m.dims.c * m.dims.t * m.dims.z);
            return s;
        };
        DatasetManifest m = manifestFromNames(lastComponent(folder.path), folder.tiffs, rule, probe, unmatched);
        m.filesFolder = folder.path;
        return m;
    }

    std::string clusterManifestName(const std::string& host, const std::string& folder) {
        std::uint64_t h = 1469598103934665603ull;   // FNV-1a
        for (const char c : host + ":" + folder) {
            h ^= static_cast<unsigned char>(c);
            h *= 1099511628211ull;
        }
        std::string name;
        for (const char c : lastComponent(folder)) {
            const bool plain = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.';
            name += plain ? c : '_';
        }
        if (name.size() > 60) name.resize(60);
        if (name.empty() || name == "." || name == "..") name = "folder";
        char hex[24];
        std::snprintf(hex, sizeof hex, "%016llx", static_cast<unsigned long long>(h));
        return name + "-" + hex + ".toml";
    }

    std::string writeClusterManifestScript(const std::string& text, const std::string& fileName) {
        // base64 on the command channel: the manifest's quotes never meet a shell
        return std::string(kPythonPrelude) + "\"$P\" - " + ssh::shellQuote(fileName) + " \"$HOME\"" +
               " <<'SIRIUS_MANIFEST'\n"
               "import base64, json, os, sys\n"
               "data = base64.b64decode('" +
               base64(text) +
               "')\n"
               "home = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else os.path.expanduser('~')\n"
               "d = os.path.join(home, '.sirius', 'manifests')\n"
               "try:\n"
               "    if not os.path.isdir(d):\n"
               "        os.makedirs(d, 0o700)\n"
               "    p = os.path.join(d, os.path.basename(sys.argv[1]))\n"
               "    with open(p + '.part', 'wb') as f:\n"
               "        f.write(data)\n"
               "    os.replace(p + '.part', p)\n"
               "    out = {'path': p}\n"
               "except OSError as x:\n"
               "    out = {'error': 'cannot write the manifest into %s: %s' % (d, x.strerror or x)}\n"
               "sys.stdout.write(json.dumps(out) + '\\n')\n"
               "SIRIUS_MANIFEST\n";
    }

    std::string writeClusterManifest(cluster::Session& session, const ClusterFolder& folder, DatasetManifest manifest) {
        manifest.filesFolder = folder.path;
        const std::string script = writeClusterManifestScript(manifest.toText(), clusterManifestName(folder.host, folder.path));
        const ssh::CommandResult r = session.run(script, std::chrono::seconds(120));
        const json j = lastJsonLine(r.out + "\n" + r.err, "writing the dataset's manifest");
        if (j.contains("error")) throw ssh::SshError(j["error"].is_string() ? j["error"].get<std::string>() : "cannot write the manifest");
        const std::string path = clusterSlashes(j.value("path", std::string()));
        if (path.empty()) throw ssh::SshError("the cluster did not say where it wrote the manifest", r.err);
        return makeClusterPath(folder.host, path);
    }

} // namespace sirius::app
