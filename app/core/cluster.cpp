#include "core/cluster.hpp"

#include "core/build_info.hpp"
#include "core/cancel.hpp"
#include "core/errors.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <sstream>

#include "sirius_worker_script_generated.hpp"

namespace sirius::app::cluster {

    using json = nlohmann::json;
    using ssh::shellQuote;
    using ssh::remotePathWord;

    // --- the profile -------------------------------------------------------------------

    void Profile::remember() {
        if (!host.empty()) perHost[host] = SlurmChoice{partition, account, qos, time, bind, containerPythonPath};
    }

    bool Profile::recall(const std::string& h) {
        const auto it = perHost.find(h);
        if (it == perHost.end()) return false;
        partition = it->second.partition;
        account = it->second.account;
        qos = it->second.qos;
        time = it->second.time;
        if (it->second.bind) bind = *it->second.bind;
        if (it->second.containerPythonPath) containerPythonPath = *it->second.containerPythonPath;
        return true;
    }

    const char* stepTitle(Step s) {
        switch (s) {
            case Step::Login: return "Log in";
            case Step::Submit: return "Ask for a job";
            case Step::Queue: return "Wait for the job to start";
            case Step::Checks: return "Check the image and the engine in the job";
            case Step::Start: return "Start the worker in the job";
            case Step::Hello: return "Connect to the worker";
        }
        return "";
    }

    // --- listing -------------------------------------------------------------------------

    std::string listingScript(const std::string& path, int maxEntries) {
        const std::string word = path.empty() ? std::string("\"$HOME\"") : remotePathWord(path);
        return "P=$(command -v python3 || command -v python) || { echo '{\"error\": \"no python3 on this host to list folders "
               "with\"}'; exit 0; }\n"
               "\"$P\" - " +
               word + " " + std::to_string(maxEntries) + " <<'SIRIUS_LS'\n"
                                                         "import json, os, sys\n"
                                                         "def clean(s):\n"
                                                         "    try:\n"
                                                         "        s.encode('utf-8')\n"
                                                         "        return s\n"
                                                         "    except UnicodeEncodeError:\n"
                                                         "        return s.encode('utf-8', 'surrogateescape').decode('utf-8', 'replace')\n"
                                                         "p = os.path.abspath(sys.argv[1])\n"
                                                         "cap = int(sys.argv[2])\n"
                                                         "out = {'path': clean(p), 'home': clean(os.path.expanduser('~')), 'entries': [], 'truncated': False}\n"
                                                         "try:\n"
                                                         "    with os.scandir(p) as it:\n"
                                                         "        for e in it:\n"
                                                         "            if len(out['entries']) >= cap:\n"
                                                         "                out['truncated'] = True\n"
                                                         "                break\n"
                                                         "            try:\n"
                                                         "                link = e.is_symlink()\n"
                                                         "                d = e.is_dir()\n"
                                                         "            except OSError:\n"
                                                         "                link, d = True, False\n"
                                                         "            try:\n"
                                                         "                st = e.stat()\n"
                                                         "            except OSError:\n"
                                                         "                st = None\n"
                                                         "            out['entries'].append([clean(e.name), int(d), 0 if (d or st is None) else int(st.st_size),\n"
                                                         "                                   0 if st is None else int(st.st_mtime), int(link)])\n"
                                                         "except OSError as e:\n"
                                                         "    out = {'error': '%s: %s' % (clean(p), e.strerror or e)}\n"
                                                         "print(json.dumps(out))\n"
                                                         "SIRIUS_LS\n";
    }

    Listing parseListing(const std::string& line) {
        json j;
        try {
            j = json::parse(line);
        } catch (const json::exception&) {
            throw ssh::SshError("the cluster's answer to a folder listing was not readable", line.substr(0, 400));
        }
        if (j.contains("error")) throw ssh::SshError(j["error"].is_string() ? j["error"].get<std::string>() : "cannot list the folder");
        Listing l;
        l.path = j.value("path", std::string());
        l.home = j.value("home", std::string());
        l.truncated = j.value("truncated", false);
        if (j.contains("entries") && j["entries"].is_array())
            for (const json& e : j["entries"]) {
                if (!e.is_array() || e.size() < 5 || !e[0].is_string()) continue;
                Entry en;
                en.name = e[0].get<std::string>();
                en.dir = e[1].is_number() && e[1].get<int>() != 0;
                en.size = e[2].is_number() ? e[2].get<std::uint64_t>() : 0;
                en.mtime = e[3].is_number() ? e[3].get<double>() : 0.0;
                en.link = e[4].is_number() && e[4].get<int>() != 0;
                l.entries.push_back(std::move(en));
            }
        auto lower = [](std::string s) {
            for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            return s;
        };
        std::stable_sort(l.entries.begin(), l.entries.end(), [&](const Entry& a, const Entry& b) {
            if (a.dir != b.dir) return a.dir;
            return lower(a.name) < lower(b.name);
        });
        return l;
    }

    // --- what the cluster offers -----------------------------------------------------------

    namespace {
        std::string stripped(const std::string& s) {
            std::size_t a = 0, b = s.size();
            while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
            while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
            return s.substr(a, b - a);
        }

        std::vector<std::string> splitOn(const std::string& s, char sep) {
            std::vector<std::string> out;
            std::string cur;
            for (char c : s) {
                if (c == sep) {
                    out.push_back(cur);
                    cur.clear();
                } else {
                    cur.push_back(c);
                }
            }
            out.push_back(cur);
            return out;
        }

        // The leading digits of "64", "32+", "250000+"; 0 when there are none.
        long long leadingNumber(const std::string& s) {
            long long v = 0;
            std::size_t i = 0;
            while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])) && v < (1LL << 50)) v = v * 10 + (s[i++] - '0');
            return v;
        }

        bool allDigits(const std::string& s) {
            if (s.empty()) return false;
            for (char c : s)
                if (!std::isdigit(static_cast<unsigned char>(c))) return false;
            return true;
        }

        std::string lowerCase(std::string s) {
            for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            return s;
        }

        std::string upperCase(std::string s) {
            for (char& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            return s;
        }

        // The GPUs of one sinfo %G: "gpu:a100:8(S:0-1),shard:..." -> 8, "a100".
        void gresGpus(const std::string& gres, int& count, std::string& type) {
            count = 0;
            type.clear();
            // items are comma separated, but a socket list "(S:0,1)" has commas of its own
            std::vector<std::string> items;
            std::string cur;
            int depth = 0;
            for (char c : gres) {
                if (c == '(') ++depth;
                else if (c == ')' && depth > 0) --depth;
                if (c == ',' && depth == 0) {
                    items.push_back(cur);
                    cur.clear();
                } else {
                    cur.push_back(c);
                }
            }
            items.push_back(cur);
            for (std::string item : items) {
                const std::size_t paren = item.find('(');
                if (paren != std::string::npos) item = item.substr(0, paren);
                item = stripped(item);
                if (item.rfind("gpu", 0) != 0) continue;
                const std::vector<std::string> f = splitOn(item, ':');
                if (f.empty() || f[0] != "gpu") continue;
                int n = 0;
                std::string t;
                if (f.size() == 2) {
                    if (allDigits(f[1])) n = static_cast<int>(leadingNumber(f[1]));
                    else t = f[1];
                } else if (f.size() >= 3) {
                    t = f[1];
                    n = static_cast<int>(leadingNumber(f[2]));
                } else {
                    n = 1;
                }
                count += n;
                if (type.empty()) type = t;
            }
        }

        // One "@@name" section of the script's output and its exit code.
        struct Section {
            std::vector<std::string> lines;
            int rc = -1;
            bool seen = false;
            std::string text() const {
                std::string s;
                for (const std::string& l : lines) s += (s.empty() ? "" : "\n") + l;
                return stripped(s);
            }
        };

        // "sacctmgr: error: ..." made into one short line for a note
        std::string firstLine(const std::string& s) {
            const std::string t = stripped(s);
            const std::size_t nl = t.find('\n');
            std::string l = nl == std::string::npos ? t : t.substr(0, nl);
            if (l.size() > 200) l = l.substr(0, 200) + "\xE2\x80\xA6";
            return l;
        }
    } // namespace

    // --- the container's binds --------------------------------------------------------------

    std::vector<std::string> bindHostPaths(const std::string& bind) {
        std::vector<std::string> out;
        for (const std::string& entry : splitOn(bind, ',')) {
            const std::string e = stripped(entry);
            const std::string host = stripped(e.substr(0, e.find(':')));
            if (!host.empty()) out.push_back(host);
        }
        return out;
    }

    std::vector<std::pair<std::string, std::string>> bindPairs(const std::string& bind) {
        std::vector<std::pair<std::string, std::string>> out;
        for (const std::string& entry : splitOn(bind, ',')) {
            const std::vector<std::string> f = splitOn(stripped(entry), ':');
            const std::string host = f.empty() ? std::string() : stripped(f[0]);
            if (host.empty()) continue;
            const std::string inside = f.size() > 1 && !stripped(f[1]).empty() ? stripped(f[1]) : host;
            out.emplace_back(host, inside);
        }
        return out;
    }

    std::string emptyBindWarning(const Profile& p) {
        if (stripped(p.container).empty() || !bindHostPaths(p.bind).empty()) return {};
        return "the image sees only itself and your home folder: datasets in other folders will not open "
               "\xE2\x80\x94 add those folders under Data folders";
    }

    namespace {
        // "~" and "~/x" with `home` for it, without trailing slashes ("/" stays).
        std::string expandedPath(const std::string& path, const std::string& home) {
            std::string p = stripped(path);
            if (p == "~") p = home;
            else if (p.rfind("~/", 0) == 0) p = home + p.substr(1);
            while (p.size() > 1 && p.back() == '/') p.pop_back();
            return p;
        }

        bool isWithin(const std::string& path, const std::string& root) {
            if (root.empty() || root.front() != '/') return false;
            if (root == "/") return true;
            return path == root || (path.size() > root.size() && path.compare(0, root.size(), root) == 0 && path[root.size()] == '/');
        }
    } // namespace

    std::string unboundPathMessage(const Profile& p, const std::string& home, const std::string& remotePath) {
        const std::string h = expandedPath(home, {});
        if (stripped(p.container).empty() || h.empty() || h.front() != '/') return {};
        const std::string path = expandedPath(remotePath, h);
        if (path.empty() || path.front() != '/') return {};
        // what apptainer shows by itself (the home folder, /tmp), what the
        // job binds (the checkout, the run folder in the home), and the binds
        std::vector<std::string> roots = {h, "/tmp", expandedPath(p.checkout, h)};
        if (p.engine && !stripped(p.scratch).empty()) roots.push_back(expandedPath(p.scratch, h));
        for (const std::string& b : bindHostPaths(p.bind)) roots.push_back(expandedPath(b, h));
        for (const std::string& r : roots)
            if (isWithin(path, r)) return {};
        const std::size_t second = path.find('/', 1);
        const std::string top = second == std::string::npos ? path : path.substr(0, second);
        return "This path is not bound into the worker's container: add its folder under Data folders (e.g. " + top + ") and reconnect.";
    }

    std::string clusterInfoScript() {
        // Fixed text only: nothing of the profile is in it. $u is the
        // cluster's own name for this user; `timeout` keeps a slurmdbd that
        // does not answer from holding the command channel.
        return "u=\"${USER:-$(id -un 2>/dev/null)}\"\n"
               "T=; command -v timeout >/dev/null 2>&1 && T='timeout 30'\n"
               "echo \"@@user $u\"\n"
               "echo \"@@home $HOME\"\n"
               "echo @@sinfo; $T sinfo -h -o '%P|%a|%l|%D|%t|%G|%c|%m' 2>&1; echo \"@@rc $?\"\n"
               "echo @@assoc; $T sacctmgr -n -P show assoc user=\"$u\" format=partition,account,qos,defaultqos 2>&1; echo \"@@rc $?\"\n"
               "echo @@qos; $T sacctmgr -n -P show qos format=name,maxwall 2>&1; echo \"@@rc $?\"\n"
               "echo @@scontrol; $T scontrol -o show partition 2>&1; echo \"@@rc $?\"\n"
               "echo @@end\n";
    }

    ClusterInfo parseClusterInfo(const std::string& output) {
        ClusterInfo info;
        std::map<std::string, Section> sections;
        std::string current;
        {
            std::istringstream in(output);
            std::string line;
            while (std::getline(in, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.rfind("@@user", 0) == 0) {
                    info.user = stripped(line.substr(6));
                    continue;
                }
                if (line.rfind("@@home", 0) == 0) {
                    info.home = stripped(line.substr(6));
                    continue;
                }
                if (line.rfind("@@rc ", 0) == 0) {
                    if (!current.empty()) {
                        sections[current].rc = static_cast<int>(leadingNumber(stripped(line.substr(5))));
                    }
                    current.clear();
                    continue;
                }
                if (line.rfind("@@", 0) == 0) {
                    current = stripped(line.substr(2));
                    if (current == "end") current.clear();
                    else sections[current].seen = true;
                    continue;
                }
                if (!current.empty()) sections[current].lines.push_back(line);
            }
        }
        auto answered = [&](const char* name) {
            const auto it = sections.find(name);
            return it != sections.end() && it->second.seen && it->second.rc == 0;
        };
        auto why = [&](const char* name) {
            const auto it = sections.find(name);
            if (it == sections.end() || !it->second.seen) return std::string("no answer");
            if (it->second.rc == 127) return std::string("not installed");
            if (it->second.rc == 124) return std::string("no answer within 30 s");
            const std::string t = firstLine(it->second.text());
            return t.empty() ? "exit " + std::to_string(it->second.rc) : t;
        };

        // sinfo: one line for each group of alike nodes of a partition
        if (!answered("sinfo")) {
            info.error = "sinfo did not list the partitions (" + why("sinfo") + ")";
        } else {
            std::map<std::string, std::size_t> index;
            for (const std::string& raw : sections["sinfo"].lines) {
                const std::vector<std::string> f = splitOn(stripped(raw), '|');
                if (f.size() < 8) continue;
                std::string name = stripped(f[0]);
                bool isDefault = false;
                if (!name.empty() && name.back() == '*') {
                    isDefault = true;
                    name.pop_back();
                }
                if (name.empty()) continue;
                auto it = index.find(name);
                if (it == index.end()) {
                    it = index.emplace(name, info.partitions.size()).first;
                    Partition p;
                    p.name = name;
                    info.partitions.push_back(p);
                }
                Partition& p = info.partitions[it->second];
                p.isDefault = p.isDefault || isDefault;
                p.up = stripped(f[1]) == "up";
                p.maxTime = stripped(f[2]);
                const int n = static_cast<int>(leadingNumber(stripped(f[3])));
                p.nodes += n;
                // the state without its flags: "idle~" (powered down), "mix*" (not responding)
                const std::string st = lowerCase(stripped(f[4]));
                std::string base;
                for (char c : st)
                    if (std::isalpha(static_cast<unsigned char>(c))) base.push_back(c);
                const bool responding = st.find('*') == std::string::npos;
                if (responding && base == "idle") p.idle += n;
                else if (responding && base == "mix") p.mixed += n;
                int gpus = 0;
                std::string type;
                gresGpus(stripped(f[5]), gpus, type);
                if (gpus > p.gpusPerNode) {
                    p.gpusPerNode = gpus;
                    p.gpuType = type;
                }
                p.cpusPerNode = std::max(p.cpusPerNode, static_cast<int>(leadingNumber(stripped(f[6]))));
                p.memPerNodeMB = std::max(p.memPerNodeMB, leadingNumber(stripped(f[7])));
            }
        }

        // sacctmgr: the user's associations, "partition|account|qos,qos|defaultqos"
        if (answered("assoc")) {
            info.associationsKnown = true;
            for (const std::string& raw : sections["assoc"].lines) {
                const std::vector<std::string> f = splitOn(stripped(raw), '|');
                if (f.size() < 3) continue;
                Association a;
                a.partition = stripped(f[0]);
                a.account = stripped(f[1]);
                for (const std::string& q : splitOn(f[2], ','))
                    if (!stripped(q).empty()) a.qos.push_back(stripped(q));
                if (f.size() >= 4) a.defaultQos = stripped(f[3]);
                if (a.account.empty()) continue;
                info.associations.push_back(std::move(a));
            }
            if (info.associations.empty())
                info.notes.push_back("sacctmgr lists no association for " + (info.user.empty() ? std::string("this user") : info.user) +
                                     ": sbatch may refuse every partition");
        } else {
            info.notes.push_back("sacctmgr did not list your associations (" + why("assoc") + "): every partition is shown as usable");
        }
        if (answered("qos")) {
            for (const std::string& raw : sections["qos"].lines) {
                const std::vector<std::string> f = splitOn(stripped(raw), '|');
                if (f.size() < 2 || stripped(f[0]).empty()) continue;
                info.qosMaxWall[stripped(f[0])] = stripped(f[1]);
            }
        }
        // scontrol: whole-node partitions
        if (answered("scontrol")) {
            info.exclusiveKnown = true;
            for (const std::string& raw : sections["scontrol"].lines) {
                std::istringstream words(raw);
                std::string w, name;
                bool exclusive = false;
                while (words >> w) {
                    const std::size_t eq = w.find('=');
                    if (eq == std::string::npos) continue;
                    const std::string key = w.substr(0, eq), value = upperCase(w.substr(eq + 1));
                    if (key == "PartitionName") name = w.substr(eq + 1);
                    else if ((key == "OverSubscribe" || key == "Shared") && value.rfind("EXCLUSIVE", 0) == 0) exclusive = true;
                }
                for (Partition& p : info.partitions)
                    if (p.name == name) p.exclusive = exclusive;
            }
        }
        return info;
    }

    const Partition* findPartition(const ClusterInfo& info, const std::string& name) {
        for (const Partition& p : info.partitions)
            if (p.name == name) return &p;
        return nullptr;
    }

    namespace {
        // The associations that let the user submit to `partition`: its own first, then those for every partition.
        std::vector<const Association*> associationsFor(const ClusterInfo& info, const std::string& partition) {
            std::vector<const Association*> out;
            for (const Association& a : info.associations)
                if (a.partition == partition) out.push_back(&a);
            for (const Association& a : info.associations)
                if (a.partition.empty()) out.push_back(&a);
            return out;
        }

        void addOnce(std::vector<std::string>& v, const std::string& s) {
            if (!s.empty() && std::find(v.begin(), v.end(), s) == v.end()) v.push_back(s);
        }
    } // namespace

    bool hasAssociation(const ClusterInfo& info, const std::string& partition) {
        return !info.associationsKnown || !associationsFor(info, partition).empty();
    }

    std::vector<std::string> accountsFor(const ClusterInfo& info, const std::string& partition) {
        std::vector<std::string> out;
        for (const Association* a : associationsFor(info, partition)) addOnce(out, a->account);
        return out;
    }

    std::vector<std::string> qosFor(const ClusterInfo& info, const std::string& partition, const std::string& account) {
        std::vector<std::string> out;
        for (const Association* a : associationsFor(info, partition))
            if (a->account == account) {
                addOnce(out, a->defaultQos);
                for (const std::string& q : a->qos) addOnce(out, q);
            }
        return out;
    }

    std::string partitionSummary(const Partition& p, bool full) {
        const std::string dot = " \xC2\xB7 ";
        std::string s;
        if (!p.up) s += "down" + dot;
        s += std::to_string(p.nodes) + (p.nodes == 1 ? " node" : " nodes");
        std::string free;
        if (p.idle > 0) free = std::to_string(p.idle) + " idle";
        if (p.mixed > 0) free += (free.empty() ? "" : ", ") + std::to_string(p.mixed) + " partly used";
        s += " (" + (free.empty() ? std::string("none free") : free) + ")";
        if (p.gpusPerNode > 0)
            s += dot + std::to_string(p.gpusPerNode) + "x " + (p.gpuType.empty() ? std::string("GPU") : upperCase(p.gpuType)) + " per node";
        else
            s += dot + "no GPUs listed";
        if (full) {
            if (p.cpusPerNode > 0) s += dot + std::to_string(p.cpusPerNode) + " CPUs";
            if (p.memPerNodeMB > 0) s += dot + std::to_string(p.memPerNodeMB / 1024) + "G";
        }
        if (!p.maxTime.empty()) {
            const long long t = slurmTimeSeconds(p.maxTime);
            s += dot + (t == -1 ? std::string("no time limit") : "max " + p.maxTime);
        }
        if (p.isDefault) s += dot + "default";
        return s;
    }

    std::string shortGpuName(const std::string& name) {
        std::string s = name;
        s.erase(0, std::min(s.size(), s.find_first_not_of(" \t")));
        for (const char* vendor : {"NVIDIA ", "Tesla "})
            if (s.rfind(vendor, 0) == 0) s = s.substr(std::char_traits<char>::length(vendor));
        // "A100-SXM4-80GB": the model before the form factor
        if (const std::size_t dash = s.find('-'); dash != std::string::npos && dash > 0) s = s.substr(0, dash);
        // "A100 80GB PCIe": the memory is said from memory_mb, the bus is not of interest
        std::istringstream words(s);
        std::string out, w;
        while (words >> w) {
            const std::string l = lowerCase(w);
            const bool memory = l.size() > 2 && l.compare(l.size() - 2, 2, "gb") == 0 &&
                                std::all_of(l.begin(), l.end() - 2, [](char c) { return std::isdigit(static_cast<unsigned char>(c)) != 0; });
            if (memory || l == "pcie" || l == "nvl" || l.rfind("sxm", 0) == 0 || l.rfind("hbm", 0) == 0) continue;
            out += (out.empty() ? "" : " ") + w;
        }
        return out.empty() ? name : out;
    }

    std::string gpuSummary(const std::vector<GpuInfo>& gpus) {
        // identical GPUs counted together, in the order they come
        std::vector<std::pair<std::string, int>> kinds;
        for (const GpuInfo& g : gpus) {
            std::string kind = shortGpuName(g.name);
            if (kind.empty()) kind = "GPU";
            if (g.memoryMb > 0) kind += " " + std::to_string((g.memoryMb + 512) / 1024) + " GB";
            auto it = std::find_if(kinds.begin(), kinds.end(), [&](const auto& k) { return k.first == kind; });
            if (it == kinds.end()) kinds.emplace_back(kind, 1);
            else ++it->second;
        }
        std::string s;
        for (const auto& [kind, n] : kinds) s += (s.empty() ? "" : " + ") + std::to_string(n) + "\xC3\x97 " + kind;
        return s;
    }

    std::string shortNodeName(const std::string& node) {
        const std::size_t dot = node.find('.');
        if (dot == std::string::npos || dot == 0) return node;
        const std::string first = node.substr(0, dot);
        // an address keeps its dots
        if (std::all_of(first.begin(), first.end(), [](char c) { return std::isdigit(static_cast<unsigned char>(c)) != 0; })) return node;
        return first;
    }

    bool gpuUsable(const WorkerCapabilities& caps) { return caps.cuda || caps.cudaUsable; }

    bool hasEngine(const WorkerCapabilities& caps) { return caps.engine.is_object(); }

    std::string gpuUnusableReason(const std::string& node, const WorkerCapabilities& caps) {
        if (gpuUsable(caps)) return {};
        if (!caps.gpus.empty())
            return node + " has " + gpuSummary(caps.gpus) + ", but the worker cannot compute on it: " +
                   (caps.cudaReason.empty() ? std::string("it reports no CUDA") : caps.cudaReason);
        return "The worker job on " + node + " has no GPU (it reports " + caps.device + "): reconnect with GPUs \xE2\x89\xA5 1 to use one";
    }

    std::string unusableGpuNote(const WorkerCapabilities& caps) {
        if (gpuUsable(caps) || caps.gpus.empty()) return {};
        return gpuSummary(caps.gpus) + " not usable: " + (caps.cudaReason.empty() ? std::string("the worker reports no CUDA") : caps.cudaReason);
    }

    std::vector<NodeDevice> nodeDevices(const std::string& node, const WorkerCapabilities& caps) {
        const std::string dot = " \xC2\xB7 ";
        const std::string name = shortNodeName(node.empty() ? caps.hostname : node);
        const std::string at = name.empty() ? std::string() : name + dot;
        NodeDevice gpu;
        gpu.gpu = true;
        gpu.usable = gpuUsable(caps);
        if (!caps.gpus.empty()) gpu.label = at + gpuSummary(caps.gpus);
        else if (caps.device.rfind("cuda", 0) == 0) gpu.label = at + caps.device;   // a worker older than "gpus"
        else gpu.label = at + (gpu.usable ? "GPU" : "no GPU");
        if (!gpu.usable)
            gpu.why = !caps.cudaReason.empty() ? caps.cudaReason
                      : caps.gpus.empty()      ? std::string("the job has no GPU: reconnect with GPUs \xE2\x89\xA5 1")
                                               : std::string("the worker reports no CUDA");
        int threads = caps.cpuThreads;
        if (threads <= 0 && caps.engine.is_object() && caps.engine.contains("cpu_threads") && caps.engine["cpu_threads"].is_number_integer())
            threads = caps.engine["cpu_threads"].get<int>();
        NodeDevice cpu;
        cpu.label = at + "CPU" + (threads > 0 ? dot + std::to_string(threads) + (threads == 1 ? " thread" : " threads") : std::string());
        return {gpu, cpu};
    }

    std::string partitionWarning(const Partition& p) {
        // Read from scontrol (OverSubscribe=EXCLUSIVE); the DGX note is by
        // name, since no Slurm setting says that a node's GPUs are not
        // isolated per job (fiona's dgx: a job sees all of them, and the
        // group shares the node).
        const std::string gpus = p.gpusPerNode > 0 ? std::to_string(p.gpusPerNode) + " " + (p.gpuType.empty() ? std::string("GPUs") : upperCase(p.gpuType) + "s")
                                                   : std::string("GPUs");
        if (lowerCase(p.name).find("dgx") != std::string::npos)
            return "Shared DGX: the GPUs are not isolated per job. The worker takes the whole node, all " + gpus +
                   ", from everyone in the group who shares it, for as long as it runs: keep the time short and Disconnect with "
                   "\"Cancel the job\" when done.";
        if (p.exclusive)
            return "Whole nodes only (OverSubscribe=EXCLUSIVE): the worker takes a node to itself, all " + gpus +
                   " of it, whatever GPUs you ask for; it may wait longer for one to be free.";
        return {};
    }

    long long slurmTimeSeconds(const std::string& text) {
        const std::string t = lowerCase(stripped(text));
        if (t == "infinite" || t == "unlimited") return -1;
        if (t.empty()) return -2;
        long long days = 0;
        std::string rest = t;
        const std::size_t dash = t.find('-');
        if (dash != std::string::npos) {
            if (!allDigits(t.substr(0, dash))) return -2;
            days = leadingNumber(t.substr(0, dash));
            rest = t.substr(dash + 1);
        }
        const std::vector<std::string> f = splitOn(rest, ':');
        for (const std::string& x : f)
            if (!allDigits(x) || x.size() > 6) return -2;
        std::vector<long long> v;
        for (const std::string& x : f) v.push_back(leadingNumber(x));
        long long s = 0;
        if (dash != std::string::npos) {
            // days-hours[:minutes[:seconds]]
            if (v.size() > 3) return -2;
            s = v[0] * 3600 + (v.size() > 1 ? v[1] * 60 : 0) + (v.size() > 2 ? v[2] : 0);
        } else if (v.size() == 1) {
            s = v[0] * 60;   // minutes
        } else if (v.size() == 2) {
            s = v[0] * 60 + v[1];   // minutes:seconds
        } else if (v.size() == 3) {
            s = v[0] * 3600 + v[1] * 60 + v[2];
        } else {
            return -2;
        }
        return days * 86400 + s;
    }

    std::string slurmTimeText(long long seconds) {
        if (seconds < 0) return "infinite";
        const long long d = seconds / 86400, h = (seconds / 3600) % 24, m = (seconds / 60) % 60, s = seconds % 60;
        char buf[48];
        if (d > 0) std::snprintf(buf, sizeof buf, "%lld-%02lld:%02lld:%02lld", d, h, m, s);
        else std::snprintf(buf, sizeof buf, "%02lld:%02lld:%02lld", h, m, s);
        return buf;
    }

    long long memoryMB(const std::string& text) {
        std::string t = upperCase(stripped(text));
        if (!t.empty() && t.back() == 'B') t.pop_back();
        if (t.empty()) return -1;
        long long scale = 1;
        switch (t.back()) {
            case 'K': scale = -1024; break;
            case 'M': scale = 1; break;
            case 'G': scale = 1024; break;
            case 'T': scale = 1024 * 1024; break;
            default: break;
        }
        if (std::isalpha(static_cast<unsigned char>(t.back()))) t.pop_back();
        if (!allDigits(t)) return -1;
        const long long v = leadingNumber(t);
        return scale < 0 ? v / -scale : v * scale;
    }

    std::vector<std::string> choosePartition(Profile& p, const ClusterInfo& info, const std::string& partition) {
        std::vector<std::string> changed;
        p.partition = partition;
        const Partition* part = findPartition(info, partition);
        // the account and QoS of an association with it
        const std::vector<std::string> accounts = accountsFor(info, partition);
        if (!accounts.empty()) {
            if (std::find(accounts.begin(), accounts.end(), p.account) == accounts.end()) {
                p.account = accounts.front();
                changed.push_back("account " + p.account);
            }
            const std::vector<std::string> qos = qosFor(info, partition, p.account);
            if (qos.empty()) {
                if (!p.qos.empty()) changed.push_back("no QoS (the account's default)");
                p.qos.clear();
            } else if (std::find(qos.begin(), qos.end(), p.qos) == qos.end()) {
                p.qos = qos.front();   // the association's default QoS comes first
                changed.push_back("QoS " + p.qos);
            }
        }
        // the time within the partition's and the QoS's limits
        long long limit = -1;
        if (part) limit = slurmTimeSeconds(part->maxTime);
        if (const auto q = info.qosMaxWall.find(p.qos); q != info.qosMaxWall.end()) {
            const long long w = slurmTimeSeconds(q->second);
            if (w >= 0 && (limit < 0 || w < limit)) limit = w;
        }
        const long long want = slurmTimeSeconds(p.time);
        if (limit >= 0 && want >= 0 && want > limit) {
            p.time = slurmTimeText(limit);
            changed.push_back("time " + p.time + " (the most allowed)");
        }
        if (part) {
            // Nodes that list no GPUs leave the count as it is: a GPU that
            // Slurm does not manage (a shared DGX) is the user's to ask for.
            if (part->gpusPerNode > 0 && p.gpus > part->gpusPerNode) {
                p.gpus = part->gpusPerNode;
                changed.push_back(std::to_string(p.gpus) + (p.gpus == 1 ? " GPU" : " GPUs") + " (a node's)");
            }
            if (part->cpusPerNode > 0 && p.cpus > part->cpusPerNode) {
                p.cpus = part->cpusPerNode;
                changed.push_back(std::to_string(p.cpus) + " CPUs (a node's)");
            }
            const long long mem = memoryMB(p.mem);
            if (part->memPerNodeMB > 0 && mem > part->memPerNodeMB) {
                p.mem = part->memPerNodeMB >= 1024 ? std::to_string(part->memPerNodeMB / 1024) + "G" : std::to_string(part->memPerNodeMB) + "M";
                changed.push_back("memory " + p.mem + " (a node's)");
            }
        }
        return changed;
    }

    std::vector<std::string> fillFromCluster(Profile& p, const ClusterInfo& info) {
        std::vector<std::string> filled;
        if (stripped(p.checkout).empty() && !info.home.empty() && info.home.front() == '/') {
            p.checkout = info.home + "/sirius";
            filled.push_back("the checkout " + p.checkout);
        }
        if (!info.error.empty() || info.partitions.empty()) return filled;
        if (stripped(p.partition).empty()) {
            const Partition* def = nullptr;
            for (const Partition& part : info.partitions)
                if (part.isDefault) {
                    def = &part;
                    break;
                }
            if (def) {
                p.partition = def->name;
                filled.push_back("partition " + p.partition + " (the cluster's default)");
                // a default partition without GPUs while others have them: ask for none
                bool gpusElsewhere = false;
                for (const Partition& part : info.partitions) gpusElsewhere = gpusElsewhere || part.gpusPerNode > 0;
                if (def->gpusPerNode == 0 && gpusElsewhere && p.gpus > 0) {
                    p.gpus = 0;
                    filled.push_back("no GPU (the default partition has none: pick a GPU partition for one)");
                }
            }
        }
        if (stripped(p.partition).empty()) return filled;
        // the account and QoS of the association with it, as a pick from the list does
        if (stripped(p.account).empty()) {
            const std::vector<std::string> accounts = accountsFor(info, p.partition);
            if (!accounts.empty()) {
                p.account = accounts.front();
                filled.push_back("account " + p.account);
            }
        }
        if (stripped(p.qos).empty() && !stripped(p.account).empty()) {
            const std::vector<std::string> qos = qosFor(info, p.partition, p.account);
            if (!qos.empty()) {
                p.qos = qos.front();
                filled.push_back("QoS " + p.qos);
            }
        }
        return filled;
    }

    // --- the title bar's button ---------------------------------------------------------

    std::string durationText(long long seconds) {
        if (seconds < 0) return {};
        const long long m = (seconds + 59) / 60;
        if (m < 60) return std::to_string(m) + " min";
        const long long h = m / 60, mm = m % 60;
        if (h < 24) {
            char buf[32];
            std::snprintf(buf, sizeof buf, "%lld h %02lld min", h, mm);
            return buf;
        }
        return std::to_string(h / 24) + " d " + std::to_string(h % 24) + " h";
    }

    ConnectionBadge connectionBadge(const Status& st, bool gpu, std::chrono::steady_clock::time_point now) {
        ConnectionBadge b;
        const std::string host = st.host.empty() ? std::string("the cluster") : st.host;
        switch (st.state) {
            case State::Idle:
                b.label = "Cluster";
                b.tooltip = st.sshUp ? "Logged in to " + host + "; no worker job yet. Click to connect."
                                     : std::string("Not connected to a cluster. Click to connect.");
                return b;
            case State::JobReady: {
                if (st.noEngine) {
                    // the worker was refused for want of SIRIUS's engine: nothing can run
                    b.kind = ConnectionBadge::Kind::Failed;
                    b.label = shortNodeName(st.node) + " \xC2\xB7 no engine";
                    b.tooltip = "Job " + st.jobId + " holds " + st.node + " on " + host + ", but there is no SIRIUS engine in it: nothing runs on the cluster.\n" +
                                st.reason + (st.fix.empty() ? std::string() : "\n" + st.fix) + "\nClick to fix.";
                    return b;
                }
                b.kind = ConnectionBadge::Kind::JobReady;
                b.label = shortNodeName(st.node) + " \xC2\xB7 job " + st.jobId + " \xC2\xB7 no worker yet";
                b.tooltip = "Job " + st.jobId + " holds " + st.node + " on " + host + "; no worker runs in it" +
                            (st.reason.empty() ? std::string() : " (" + st.reason + ")") + ".";
                if (st.jobLimitSeconds >= 0 && st.jobStarted != std::chrono::steady_clock::time_point{}) {
                    const long long used = std::chrono::duration_cast<std::chrono::seconds>(now - st.jobStarted).count();
                    b.tooltip += "\n" + durationText(std::max(0LL, st.jobLimitSeconds - used)) + " left of " + durationText(st.jobLimitSeconds);
                }
                b.tooltip += "\nClick to start the worker.";
                return b;
            }
            case State::Starting:
            case State::Connecting: {
                b.kind = ConnectionBadge::Kind::Connecting;
                b.label = st.state == State::Starting ? "Starting worker\xE2\x80\xA6" : "Connecting\xE2\x80\xA6";
                int done = 0;
                std::string step, detail;
                for (int i = 0; i < kStepCount; ++i) {
                    const StepState& s = st.steps[static_cast<std::size_t>(i)];
                    if (s.status == StepStatus::Done || s.status == StepStatus::Warning) ++done;
                    if (s.status == StepStatus::Running) {
                        step = stepTitle(static_cast<Step>(i));
                        detail = s.detail;
                    }
                }
                b.progress = static_cast<float>(done) / static_cast<float>(kStepCount);
                b.tooltip = (st.state == State::Starting ? "Starting the worker in job " + st.jobId + " on " + host : "Connecting to " + host) +
                            (step.empty() ? std::string() : ": " + step) + (detail.empty() ? std::string() : "\n" + detail);
                return b;
            }
            case State::Connected: {
                if (!hasEngine(st.caps)) {
                    b.kind = ConnectionBadge::Kind::Failed;
                    b.label = shortNodeName(st.node) + " \xC2\xB7 no engine";
                    b.tooltip = "The worker on " + st.node + " (job " + st.jobId + ") has no SIRIUS engine: nothing runs on the cluster.\nClick to fix.";
                    return b;
                }
                b.kind = ConnectionBadge::Kind::Connected;
                b.label = shortNodeName(st.node) + " \xC2\xB7 " + (gpu ? "GPU" : "CPU");
                std::string device = gpu ? gpuSummary(st.caps.gpus) : std::string();
                if (device.empty() && gpu) device = "GPU";
                if (device.empty())
                    device = "CPU" + (st.caps.cpuThreads > 0 ? " \xC2\xB7 " + std::to_string(st.caps.cpuThreads) + " threads" : std::string());
                b.tooltip = "Connected to " + host + "\nJob " + (st.jobId.empty() ? std::string("?") : st.jobId) + " on " + st.node +
                            "\nComputing on " + device;
                if (st.jobLimitSeconds == -1) {
                    b.tooltip += "\nNo time limit";
                } else if (st.jobLimitSeconds >= 0 && st.jobStarted != std::chrono::steady_clock::time_point{}) {
                    const long long used = std::chrono::duration_cast<std::chrono::seconds>(now - st.jobStarted).count();
                    b.tooltip += "\n" + durationText(std::max(0LL, st.jobLimitSeconds - used)) + " left of " + durationText(st.jobLimitSeconds);
                }
                return b;
            }
            case State::Disconnected: break;
        }
        bool failed = false;
        for (const StepState& s : st.steps) failed = failed || s.status == StepStatus::Failed;
        if (st.dropped) {
            b.kind = ConnectionBadge::Kind::Lost;
            b.label = "Cluster: lost";
            b.tooltip = "The connection to " + host + " was lost: " + st.reason + "\nClick to connect again.";
        } else if (failed && st.reason != "Cancelled.") {
            b.kind = ConnectionBadge::Kind::Failed;
            b.label = "Cluster: failed";
            b.tooltip = "Connecting to " + host + " failed: " + st.reason + "\nClick for the details.";
        } else {
            b.label = "Cluster";
            b.tooltip = "Not connected" + (st.reason.empty() ? std::string() : " (" + st.reason + ")") + ". Click to connect.";
        }
        return b;
    }

    // --- the session --------------------------------------------------------------------

    namespace {
        // A step that could not be done: what to tell the user.
        struct Failure {
            Step step;
            std::string reason;
            std::string remote;
            std::string fix;
            bool noEngine = false;   // SIRIUS's engine is not there (Status::noEngine)
        };

        std::map<std::string, std::string> keyValues(const std::string& text) {
            std::map<std::string, std::string> kv;
            std::istringstream in(text);
            std::string line;
            while (std::getline(in, line)) {
                const std::size_t eq = line.find('=');
                if (eq != std::string::npos) kv[line.substr(0, eq)] = line.substr(eq + 1);
            }
            return kv;
        }

        std::string trim(std::string s) {
            while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
            std::size_t i = 0;
            while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
            return s.substr(i);
        }

        // A --bind list as one shell word: each entry quoted, a leading "~/"
        // made $HOME (apptainer does not expand it); "" when there is none.
        std::string bindWord(const std::string& bind) {
            std::string word;
            for (const std::string& entry : splitOn(bind, ',')) {
                const std::string e = stripped(entry);
                if (e.empty()) continue;
                word += (word.empty() ? "" : ",") + remotePathWord(e);
            }
            return word;
        }

        std::string elapsed(std::chrono::steady_clock::duration d) {
            const long long s = std::chrono::duration_cast<std::chrono::seconds>(d).count();
            char buf[32];
            if (s >= 3600) std::snprintf(buf, sizeof buf, "%lld:%02lld:%02lld", s / 3600, (s / 60) % 60, s % 60);
            else std::snprintf(buf, sizeof buf, "%lld:%02lld", s / 60, s % 60);
            return buf;
        }
    } // namespace

    struct Session::Impl {
        mutable std::mutex m;
        Status status;
        Profile profile;
        PromptFn prompt;
        std::string askpassProgram;
        std::function<void()> changed;
        std::function<void(const std::string&)> log;
        std::chrono::milliseconds queuePoll{3000}, keepAlive{15000};

        std::shared_ptr<ssh::Session> ssh;            // under m
        std::unique_ptr<ssh::AskpassServer> askpass;
        Mode mode = Mode::Both;                      // what run() does, under m
        Profile jobProfile;                          // the profile the held job was submitted with, under m
        std::string token;                           // the worker's, under m
        int workerPort = 0;                          // what the worker announced, under m
        std::string workerLog;                       // the worker step's log (a shell word), under m
        int workerSeq = 0;                           // numbers the worker steps' and builds' logs, under m
        std::string engineDir;                       // the engine build the checks picked ("" none), under m
        std::string workerDir;                       // the worker's code the checks found (a shell word), under m
        std::optional<BuildInfo> engineJson;         // the picked engine build's BUILD.json, under m
        std::string buildsHint;                      // a builds folder the checks found on the cluster ("" none), under m
        std::string codeNote;                        // where the worker's code comes from, in words, under m
        std::string reattachJob;                     // a job left running at the last disconnect, to reattach to, under m
        std::string adoptId;                         // a job of the user's to take up (adoptJob), under m
        std::string adoptedJob;                      // the job taken up so ("" none): never cancelled, under m
        std::optional<ClusterInfo> info;             // under m
        std::atomic<bool> querying{false};
        std::mutex controlMutex;
        std::unique_ptr<RemoteWorker> control;       // under controlMutex

        std::thread worker, keeper, builder;
        std::atomic<bool> buildCancel{false};
        std::atomic<bool> cancel{false}, stopKeeper{false}, connecting{false};
        std::atomic<bool> abortLogin{false}, loginActive{false};
        std::mutex keeperMutex;
        std::condition_variable keeperWake;

        // --- status ------------------------------------------------------------------
        void update(const std::function<void(Status&)>& fn) {
            {
                const std::lock_guard<std::mutex> g(m);
                fn(status);
            }
            if (changed) changed();
        }
        void say(const std::string& line) {
            if (log) log(line);
        }
        void stepState(Step s, StepStatus st, const std::string& detail) {
            update([&](Status& x) {
                x.steps[static_cast<std::size_t>(s)] = StepState{st, detail};
            });
        }
        void setState(State st, const std::string& reason = {}) {
            update([&](Status& x) {
                x.state = st;
                x.since = std::chrono::steady_clock::now();
                if (!reason.empty() || st != State::Disconnected) x.reason = reason;
            });
        }
        std::shared_ptr<ssh::Session> sshSession() const {
            const std::lock_guard<std::mutex> g(m);
            return ssh;
        }
        ssh::CommandResult remote(const std::string& script, std::chrono::milliseconds timeout = std::chrono::seconds(60)) {
            auto s = sshSession();
            if (!s || !s->isOpen()) throw ssh::SshError("the SSH connection is closed");
            return s->run(script, timeout, [this] { return cancel.load(); });
        }
        bool sleepCancellable(std::chrono::milliseconds d) {
            const auto end = std::chrono::steady_clock::now() + d;
            while (std::chrono::steady_clock::now() < end) {
                if (cancel.load()) return false;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            return !cancel.load();
        }

        // --- askpass -----------------------------------------------------------------
        std::optional<std::string> ask(const ssh::Prompt& p) {
            if (p.notifyOnly) {
                say("Cluster: " + p.text);
                return std::string();
            }
            std::optional<std::string> answer = (prompt && loginActive.load()) ? prompt(p) : std::nullopt;
            if (!answer) {
                // Stop ssh before the helper answers: ssh would send an empty
                // response for a prompt the helper gives up on.
                abortLogin.store(true);
                for (int i = 0; i < 100 && loginActive.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            return answer;
        }

        // --- what the cluster offers -----------------------------------------------------
        ClusterInfo queryInfo(const std::string& host, const std::function<bool()>& cancelled) {
            auto s = sshSession();
            if (!s || !s->isOpen()) throw ssh::SshError("not logged in to the cluster: connect first");
            querying.store(true);
            if (changed) changed();
            ClusterInfo ci;
            try {
                const ssh::CommandResult r = s->run(clusterInfoScript(), std::chrono::seconds(150), cancelled);
                ci = parseClusterInfo(r.out);
            } catch (...) {
                querying.store(false);
                if (changed) changed();
                throw;
            }
            ci.host = host;
            {
                const std::lock_guard<std::mutex> g(m);
                info = ci;
            }
            querying.store(false);
            if (changed) changed();
            if (!ci.error.empty()) say("Cluster: " + ci.error);
            else say("Cluster: " + std::to_string(ci.partitions.size()) + " partitions on " + host);
            for (const std::string& n : ci.notes) say("Cluster: " + n);
            return ci;
        }

        // After the login: the partitions, unless they are known for this
        // host already and a job is what was asked for. Never fails the
        // connect (but for a cancel).
        void listPartitions(const Profile& p, bool force) {
            {
                const std::lock_guard<std::mutex> g(m);
                if (!force && info && info->host == p.host && info->error.empty()) return;
            }
            try {
                queryInfo(p.host, [this] { return cancel.load(); });
            } catch (const std::exception& e) {
                if (cancel.load()) throw Failure{Step::Login, "Cancelled.", {}, {}};
                say(std::string("Cluster: the partitions could not be listed: ") + e.what());
            }
        }

        // --- the steps -----------------------------------------------------------------
        void login(const Profile& p) {
            stepState(Step::Login, StepStatus::Running, "ssh " + p.host);
            say("Cluster: logging in to " + p.host + "\xE2\x80\xA6");
            // A listener for this login only: closed once it is over (below).
            askpass = std::make_unique<ssh::AskpassServer>([this](const ssh::Prompt& pr) { return ask(pr); });
            ssh::Options o;
            o.program = p.sshProgram;
            o.programArgs = p.sshProgramArgs;
            o.host = p.host;
            if (!askpassProgram.empty()) o.environment = askpass->environment(askpassProgram);
            auto s = std::make_shared<ssh::Session>();
            abortLogin.store(false);
            loginActive.store(true);
            try {
                s->open(o, [this] { return cancel.load() || abortLogin.load(); }, std::chrono::minutes(10));
            } catch (const ssh::SshError& e) {
                loginActive.store(false);
                askpass->close();
                if (abortLogin.load()) throw Failure{Step::Login, "Login cancelled: nothing was sent for the prompt you closed.", {}, {}};
                if (cancel.load()) throw Failure{Step::Login, "Cancelled.", {}, {}};
                throw Failure{Step::Login, std::string("SSH login to ") + p.host + " failed: " + e.what() + ".", e.detail,
                              "Check the cluster's name (as in your ~/.ssh/config, or user@host) and your password, then press Connect again (a failed login is never retried)."};
            }
            loginActive.store(false);
            askpass->close();   // nothing is asked after the login
            {
                const std::lock_guard<std::mutex> g(m);
                ssh = s;
                status.sshUp = true;
                status.host = p.host;
            }
            stepState(Step::Login, StepStatus::Done, "logged in to " + p.host);
            say("Cluster: logged in to " + p.host);
        }

        // The shell lines that find the container launcher: the profile's,
        // after `module load` of it (or of apptainer, singularity), else
        // whichever of apptainer and singularity is there. Sets L; "" when
        // there is none. The same lines are in sirius_worker.sbatch.
        static std::string launcherScript(const Profile& p) {
            return "L=" + shellQuote(p.launcher.empty() ? std::string("apptainer") : p.launcher) +
                   "\n"
                   "if ! command -v \"$L\" >/dev/null 2>&1 && type module >/dev/null 2>&1; then\n"
                   "    module load \"$L\" >/dev/null 2>&1 || module load apptainer >/dev/null 2>&1 || module load singularity >/dev/null 2>&1\n"
                   "fi\n"
                   "if ! command -v \"$L\" >/dev/null 2>&1; then\n"
                   "    for alt in apptainer singularity; do command -v \"$alt\" >/dev/null 2>&1 && { L=$alt; break; }; done\n"
                   "fi\n"
                   "command -v \"$L\" >/dev/null 2>&1 || L=\n";
        }

        // ================================================================================
        // Step 1: the job
        // ================================================================================

        // A job that only holds the allocation: it sleeps until its time
        // limit or until it is cancelled. Nothing of SIRIUS has to be on the
        // cluster for it, and nothing secret goes with it.
        void submitHolder(const Profile& p) {
            stepState(Step::Submit, StepStatus::Running, "sbatch");
            std::string script = "umask 077\n";
            script += "mkdir -p \"$HOME/.sirius/run\" && chmod 700 \"$HOME/.sirius/run\" || exit 4\n";
            script += "command -v sbatch >/dev/null 2>&1 || { echo nosbatch=1; exit 5; }\n";
            script += "cd \"$HOME\" || exit 3\n";
            const std::string hold = "echo \"sirius: job $SLURM_JOB_ID holds $(hostname) for SIRIUS's worker\"; trap 'exit 0' TERM; "
                                     "while :; do sleep 300 & wait $!; done";
            std::string cmd = "sbatch --parsable --job-name=sirius --output=\"$HOME/.sirius/run/sirius-job-%j.log\"";
            if (!p.partition.empty()) cmd += " --partition=" + shellQuote(p.partition);
            if (!p.account.empty()) cmd += " --account=" + shellQuote(p.account);
            if (!p.qos.empty()) cmd += " --qos=" + shellQuote(p.qos);
            if (!p.time.empty()) cmd += " --time=" + shellQuote(p.time);
            cmd += p.gpus > 0 ? " --gres=gpu:" + std::to_string(p.gpus) : std::string(" --gres=none");
            if (p.cpus > 0) cmd += " --cpus-per-task=" + std::to_string(p.cpus);
            if (!p.mem.empty()) cmd += " --mem=" + shellQuote(p.mem);
            cmd += " --wrap=" + shellQuote(hold);
            script += "out=$(" + cmd + ") || { rc=$?; printf '%s\\n' \"$out\"; exit $rc; }\n";
            script += "echo \"job=$out\"\n";
            const ssh::CommandResult r = remote(script, std::chrono::seconds(60));
            auto kv = keyValues(r.out);
            if (kv["nosbatch"] == "1")
                throw Failure{Step::Submit, "There is no sbatch on " + p.host + ": connect to the cluster's login node, the one you submit jobs from.", trim(r.err),
                              "Set Cluster on the Connect page to the login node (ask your cluster's support which one that is)."};
            std::string id;
            for (char c : kv["job"]) {   // "4711" or "4711;cluster"
                if (std::isdigit(static_cast<unsigned char>(c))) id.push_back(c);
                else if (!id.empty()) break;
            }
            if (!r.ok() || id.empty())
                throw Failure{Step::Submit, "sbatch refused the job.", trim(r.err.empty() ? r.out : r.err),
                              "Check the partition, account, QoS, time limit and resources on the Job page (its lists are what the cluster offers)."};
            {
                const std::lock_guard<std::mutex> g(m);
                jobProfile = p;
                adoptedJob.clear();
            }
            update([&](Status& x) {
                x.jobId = id;
                x.adopted = false;
            });
            const std::string where = p.partition.empty() ? std::string() : " to " + p.partition;
            stepState(Step::Submit, StepStatus::Done, "job " + id + where);
            say("Cluster: submitted job " + id + where);
        }

        // A job of the user's own, taken up in place of a new one: it holds
        // what it holds (its node, its GPUs: the worker's step asks for as
        // many), and SIRIUS never cancels it.
        void takeUp(const Profile& p, const std::string& id) {
            stepState(Step::Submit, StepStatus::Running, "job " + id + " (yours)");
            const ssh::CommandResult r = remote(userJobsScript(), std::chrono::seconds(60));
            const std::vector<ClusterJob> jobs = parseUserJobs(r.out);
            const auto it = std::find_if(jobs.begin(), jobs.end(), [&id](const ClusterJob& j) { return j.id == id; });
            if (it == jobs.end())
                throw Failure{Step::Submit, "Job " + id + " is not among your running or waiting jobs on " + p.host + ".", trim(r.err.empty() ? r.out : r.err),
                              "Refresh the list of your jobs, or start a new job."};
            Profile held = p;
            held.gpus = it->gpus;
            if (it->cpus > 0) held.cpus = it->cpus;
            held.mem = it->mem;
            held.partition = it->partition;
            held.time = it->limit;
            {
                const std::lock_guard<std::mutex> g(m);
                jobProfile = held;
                adoptedJob = id;
            }
            update([&](Status& x) {
                x.jobId = id;
                x.adopted = true;
            });
            const std::string what = jobSummary(*it);
            stepState(Step::Submit, StepStatus::Done, "job " + id + " (yours) \xC2\xB7 " + it->partition + (what.empty() ? std::string() : " \xC2\xB7 " + what));
            say("Cluster: taking up your job " + id + " (" + it->name + ", " + it->partition + "): SIRIUS never cancels it");
        }

        // The job's log (the holder's), in the private ~/.sirius/run.
        static std::string jobLogPath(const std::string& id) { return "\"$HOME/.sirius/run/sirius-job-" + id + ".log\""; }

        // The worker's log, or the job's before there is a worker.
        std::string logTail(const std::string& id) {
            std::string path;
            {
                const std::lock_guard<std::mutex> g(m);
                path = workerLog;
            }
            if (path.empty()) path = jobLogPath(id);
            try {
                return trim(remote("tail -n 30 " + path + " 2>/dev/null", std::chrono::seconds(30)).out);
            } catch (const std::exception&) {
                return {};
            }
        }

        std::string finalState(const std::string& id) {
            try {
                const ssh::CommandResult r = remote("sacct -n -X -P -j " + id + " -o State,ExitCode 2>/dev/null | head -n 1", std::chrono::seconds(30));
                std::string s = trim(r.out);
                const std::size_t bar = s.find('|');
                if (bar != std::string::npos) s = s.substr(0, bar) + " (exit " + s.substr(bar + 1) + ")";
                return s.empty() ? std::string("no longer in the queue") : s;
            } catch (const ssh::SshError&) {
                return "no longer in the queue";
            }
        }

        // The job is still in the queue or running (squeue knows it).
        bool jobRunning(const std::string& id) {
            try {
                const std::string state = trim(remote("squeue -h -j " + id + " -o %T 2>/dev/null", std::chrono::seconds(30)).out);
                return state == "RUNNING" || state == "PENDING" || state == "CONFIGURING";
            } catch (const std::exception&) {
                return false;
            }
        }

        // The job's steps called `name` ("4711.0"), as squeue lists them.
        std::vector<std::string> steps(const std::string& id, const std::string& name) {
            std::vector<std::string> out;
            const ssh::CommandResult r = remote("squeue -h -s -j " + id + " -o '%i|%j' 2>/dev/null", std::chrono::seconds(30));
            std::istringstream in(r.out);
            std::string line;
            while (std::getline(in, line)) {
                const std::vector<std::string> f = splitOn(trim(line), '|');
                if (f.size() >= 2 && trim(f[1]) == name && !trim(f[0]).empty()) out.push_back(trim(f[0]));
            }
            return out;
        }

        std::string waitInQueue(const Profile& p, const std::string& id) {
            stepState(Step::Queue, StepStatus::Running, "job " + id);
            const auto t0 = std::chrono::steady_clock::now();
            for (;;) {
                // state, reason, node, time used and time limit
                const ssh::CommandResult r = remote("squeue -h -j " + id + " -o '%T|%r|%N|%M|%l' 2>/dev/null", std::chrono::seconds(30));
                const std::string line = trim(r.out);
                if (line.empty()) {
                    const std::string fin = finalState(id);
                    throw Failure{Step::Queue, "Job " + id + " ended before it started: " + fin + ".", logTail(id), {}};
                }
                const std::vector<std::string> f = splitOn(line, '|');
                const std::string state = f.empty() ? line : f[0];
                const std::string reason = f.size() > 1 ? f[1] : std::string();
                const std::string node = f.size() > 2 ? f[2] : std::string();
                update([&](Status& x) { x.jobState = state; });
                if (state == "RUNNING" && !node.empty() && node != "(null)") {
                    // the time left, from what squeue says (the profile's time when it does not)
                    const long long used = f.size() > 3 ? slurmTimeSeconds(f[3]) : -2;
                    long long limit = f.size() > 4 ? slurmTimeSeconds(f[4]) : -2;
                    if (limit == -2) limit = slurmTimeSeconds(p.time);
                    const auto started = std::chrono::steady_clock::now() - std::chrono::seconds(used >= 0 ? used : 0);
                    update([&](Status& x) {
                        x.node = node;
                        x.jobLimitSeconds = limit;
                        x.jobStarted = started;
                    });
                    stepState(Step::Queue, StepStatus::Done,
                              "job " + id + " runs on " + node + " \xC2\xB7 waited " + elapsed(std::chrono::steady_clock::now() - t0));
                    say("Cluster: job " + id + " runs on " + node);
                    return node;
                }
                std::string detail = "job " + id + " \xC2\xB7 " + state;
                if (!reason.empty() && reason != "None") detail += " (" + reason + ")";
                detail += " \xC2\xB7 " + elapsed(std::chrono::steady_clock::now() - t0);
                stepState(Step::Queue, StepStatus::Running, detail);
                if (!sleepCancellable(queuePoll)) throw Failure{Step::Queue, "Cancelled while job " + id + " waits in the queue.", {}, {}};
            }
        }

        // ================================================================================
        // Step 2: the worker in the job
        // ================================================================================

        // Checks for the worker, part one, on the login node: srun, the image
        // file, the worker's code (bound into the image from the checkout, or
        // the engine build's python/), and the engine builds, to list them and
        // pick the one that fits this application. File checks only: nothing
        // here asks whether anything runs on the login node, whose scratch
        // may be mounted noexec (the engine never runs there). What runs is
        // checked in the job (checkInJob).
        void checks(const Profile& p) {
            if (trim(p.container).empty())
                throw Failure{Step::Checks, "No worker image is set: SIRIUS's worker runs in an Apptainer/Singularity image (.sif) on the cluster.", {}, "Pick the image under Worker image (Browse lists the cluster's files), or build one with \"Build an image\". "
                                                                                                                                                         "app/python/slurm/README.md says how images are made."};
            const bool builds = p.engine && trim(p.engineBin).empty() && !trim(p.engineBuilds).empty();
            // the worker's code: the engine build's own (its commit's), else the checkout's
            if (trim(p.checkout).empty() && !builds)
                throw Failure{Step::Checks, "No SIRIUS checkout on the cluster is set, and " + p.host + " did not say where your home folder is.", {}, "Set \"SIRIUS checkout on the cluster\" under Job \xE2\x96\xB8 More options to the folder you cloned SIRIUS into there."};
            stepState(Step::Checks, StepStatus::Running, "on the login node: the image file, the worker's code, the engine builds");
            const std::string co = trim(p.checkout).empty() ? std::string() : remotePathWord(p.checkout);
            std::string script;
            script += "command -v srun >/dev/null 2>&1 && echo srun=yes || echo srun=no\n";
            if (!co.empty()) script += "[ -f " + co + "/app/python/sirius_worker/__main__.py ] && echo worker=yes || echo worker=no\n";
            script += "echo \"home=$HOME\"\n";
            script += "C=" + remotePathWord(p.container) + "\n";
            script += "if [ ! -f \"$C\" ]; then echo image=no; elif test -r \"$C\"; then echo image=yes; else echo image=unreadable; fi\n";
            if (builds) script += engineBuildsScript(p.engineBuilds);
            // where per-commit builds may be, for the fix when no builds folder is set
            if (p.engine && !builds)
                script += "for d in \"$(dirname \"$C\")/sirius-builds\" \"$(dirname \"$(dirname \"$C\")\")/sirius-builds\" \"$HOME/sirius-builds\"" +
                          (co.empty() ? std::string() : " " + co + "/../sirius-builds") +
                          "; do\n"
                          "    if ls \"$d\"/*/BUILD.json >/dev/null 2>&1; then echo \"builds_hint=$(cd \"$d\" && pwd -P)\"; break; fi\n"
                          "done\n";
            const ssh::CommandResult r = remote(script, std::chrono::seconds(120));
            auto kv = keyValues(r.out);
            const std::string ask = "Build the worker image again (\"Build an image\", or app/python/slurm/README.md), or ask whoever builds it on this "
                                    "cluster for the current one, and pick it under Worker image.";
            if (kv["srun"] != "yes")
                throw Failure{Step::Checks, "There is no srun on " + p.host + ": the worker starts in the job with srun.", trim(r.err),
                              "Connect to the cluster's login node, the one you submit jobs from."};
            if (!trim(kv["home"]).empty()) update([&](Status& x) { x.home = trim(kv["home"]); });
            if (kv["image"] == "unreadable")
                throw Failure{Step::Checks, "The worker image " + p.container + " on " + p.host + " cannot be read (test -r failed).", trim(r.err),
                              "Make it readable to you (chmod a+r), or pick a copy you can read under Worker image."};
            if (kv["image"] != "yes") throw Failure{Step::Checks, "There is no worker image at " + p.container + " on " + p.host + ".", trim(r.err), ask};
            // the engine build that serves this application, from what its files say
            std::string buildDir, buildNote, codeDir;
            std::optional<BuildInfo> buildJson;
            if (builds) {
                if (kv["builds"] != "yes")
                    throw Failure{Step::Checks, "There is no engine builds folder at " + p.engineBuilds + " on " + p.host + ".", trim(r.err),
                                  engineBuildFix(p.engineBuilds, buildInfo()), true};
                const std::vector<EngineBuild> found = parseEngineBuilds(r.out);
                const int pick = pickEngineBuild(found, buildInfo(), &buildNote);
                if (pick < 0) throw Failure{Step::Checks, buildNote, listBuilds(found), engineBuildFix(p.engineBuilds, buildInfo()), true};
                const EngineBuild& b = found[static_cast<std::size_t>(pick)];
                buildDir = trim(p.engineBuilds);
                while (buildDir.size() > 1 && buildDir.back() == '/') buildDir.pop_back();
                buildDir += "/" + b.dir;
                buildJson = b.info;
                if (b.python) codeDir = remotePathWord(buildDir + "/python");
            } else if (p.engine) {
                buildNote = trim(p.engineBin).empty() ? std::string("the image's own engine") : "engine " + trim(p.engineBin);
            }
            std::string code = "worker's code from the engine build";
            if (codeDir.empty()) {
                if (co.empty() || kv["worker"] != "yes")
                    throw Failure{Step::Checks,
                                  co.empty() ? "No SIRIUS checkout on the cluster is set, and the engine build has no python/ folder with the worker's code."
                                             : "There is no SIRIUS checkout at " + p.checkout + " on " + p.host + " (it needs app/python/sirius_worker).",
                                  trim(r.err),
                                  "Clone this SIRIUS repository there (the same version as this application), or set \"SIRIUS checkout on the cluster\" under "
                                  "Job \xE2\x96\xB8 More options to where it is."};
                codeDir = co + "/app/python";
                code = "checkout";
            }
            {
                const std::lock_guard<std::mutex> g(m);
                engineDir = buildDir;
                engineJson = buildJson;
                workerDir = codeDir;
                buildsHint = trim(kv["builds_hint"]);
                codeNote = code;
            }
            update([&](Status& x) {
                x.engineBuild = buildDir;
                x.engineBuildNote = buildNote;
            });
            stepState(Step::Checks, StepStatus::Running, code + (buildDir.empty() ? std::string() : " \xC2\xB7 engine " + shortCommit(buildDir)) + " \xC2\xB7 next: in the job");
        }

        // The last part of a build folder's path, cut to a commit's 12 characters.
        static std::string shortCommit(const std::string& buildDir) {
            const std::size_t slash = buildDir.find_last_of('/');
            return (slash == std::string::npos ? buildDir : buildDir.substr(slash + 1)).substr(0, 12);
        }

        // A worker whose code is another version than this application's.
        static std::string workerCodeFix(const Profile& p) {
            const std::string commit = buildInfo().commit.substr(0, 12);
            if (p.engine && !trim(p.engineBuilds).empty())
                return engineBuildFix(p.engineBuilds, buildInfo());
            return "Update the SIRIUS checkout on the cluster (" + (trim(p.checkout).empty() ? std::string("~/sirius") : p.checkout) +
                   ") to this application's commit " + commit + " (git fetch && git checkout " + commit +
                   "), or set Engine builds folder under Job \xE2\x96\xB8 More options so the worker's code comes with the engine build; then Restart worker.";
        }

        static std::string listBuilds(const std::vector<EngineBuild>& found) {
            std::string out;
            for (const EngineBuild& b : found) {
                std::string why;
                if (!b.binThere) why = " (no bin/sirius-cli)";
                else if (!b.runnable) why = " (bin/sirius-cli has no execute bit)";
                else if (!b.lib) why = " (no lib/)";
                out += b.dir + ": " + (b.readable ? b.info.build + why : std::string("BUILD.json unreadable")) + "\n";
            }
            return out;
        }

        // What every step of the worker's is told, through the environment of
        // the srun that starts it (sirius_worker.sbatch reads it): the
        // worker's code, the image and its launcher, the data folders, the
        // Python path, the job's GPUs, the engine and its node cache folder.
        // Never the token.
        static std::string stepEnvironment(const Profile& p, const std::string& dir, const std::string& code, int gpus) {
            std::string script = "export SIRIUS_WORKER_DIR=" + code + "\n";
            script += "export SIRIUS_CONTAINER=" + remotePathWord(p.container) + "\n";
            script += "export SIRIUS_LAUNCHER=" + shellQuote(p.launcher.empty() ? std::string("apptainer") : p.launcher) + "\n";
            // the data folders and the extra PYTHONPATH, quoted (a "~/" bind made $HOME)
            if (const std::string bw = bindWord(p.bind); !bw.empty()) script += "export SIRIUS_CONTAINER_BIND=" + bw + "\n";
            else script += "unset SIRIUS_CONTAINER_BIND\n";
            if (const std::string pp = trim(p.containerPythonPath); !pp.empty()) script += "export SIRIUS_CONTAINER_PYTHONPATH=" + shellQuote(pp) + "\n";
            else script += "unset SIRIUS_CONTAINER_PYTHONPATH\n";
            // the job's GPUs: the step is held to as many as the job asked for
            script += gpus <= 0 ? "export SIRIUS_DEVICE=cpu SIRIUS_GPUS=0\n" : "unset SIRIUS_DEVICE\nexport SIRIUS_GPUS=" + std::to_string(gpus) + "\n";
            // SIRIUS's engine (`sirius-cli serve`) as the step's main process, the Python worker its child
            if (p.engine) {
                script += "export SIRIUS_ENGINE=1\n";
                if (const std::string bin = trim(p.engineBin); !bin.empty()) {
                    script += "export SIRIUS_ENGINE_BIN=" + remotePathWord(bin) + "\nunset SIRIUS_ENGINE_DIR\n";
                } else if (!dir.empty()) {
                    // the per-commit build, bound into the image by the script
                    script += "export SIRIUS_ENGINE_DIR=" + remotePathWord(dir) + "\n";
                    script += "export SIRIUS_ENGINE_BIN=\"$SIRIUS_ENGINE_DIR/bin/sirius-cli\"\n";
                } else {
                    script += "unset SIRIUS_ENGINE_BIN SIRIUS_ENGINE_DIR\n";
                }
                if (const std::string cache = trim(p.scratch); !cache.empty()) script += "export SIRIUS_ENGINE_SCRATCH=" + remotePathWord(cache) + "\n";
                else script += "unset SIRIUS_ENGINE_SCRATCH\n";
            } else {
                script += "unset SIRIUS_ENGINE SIRIUS_ENGINE_BIN SIRIUS_ENGINE_DIR SIRIUS_ENGINE_SCRATCH\n";
            }
            return script;
        }

        // The srun that runs a step of the job: one task on the job's node,
        // beside the job's own (--overlap), given the job's GPUs.
        static std::string srunLine(const std::string& id, const std::string& name, int gpus) {
            std::string srun = "srun --jobid=" + id + " --overlap --nodes=1 --ntasks=1 --job-name=" + name;
            if (gpus > 0) srun += " --gres=gpu:" + std::to_string(gpus);
            return srun;
        }

        // Checks for the worker, part two, where it will run: the launch
        // script's --check as a step of the job, on its node, in the image,
        // with the binds, library path and environment the start gives it
        // (one srun, a few seconds). A failure stops here: nothing starts.
        void checkInJob(const Profile& p, const std::string& id, const std::string& node) {
            std::string dir, code, hint, codeText;
            std::optional<BuildInfo> buildJson;
            Profile job = p;
            {
                const std::lock_guard<std::mutex> g(m);
                dir = engineDir;
                code = workerDir;
                buildJson = engineJson;
                hint = buildsHint;
                codeText = codeNote;
                job.gpus = jobProfile.gpus;
            }
            stepState(Step::Checks, StepStatus::Running, "in job " + id + " on " + node + ": the image, the engine, the GPUs, the data folders");
            std::string script = "umask 077\n";
            script += "mkdir -p \"$HOME/.sirius/run\" && chmod 700 \"$HOME/.sirius/run\" || exit 4\n";
            // the application's launch script, as for the start (never the checkout's)
            script += uploadLaunchScript();
            script += "cd \"$HOME\" || exit 3\n";
            script += stepEnvironment(p, dir, code, job.gpus);
            script += "unset SIRIUS_TOKEN SIRIUS_TOKEN_FILE\n";
            std::string pairs;
            for (const auto& [host, inside] : bindPairs(p.bind)) pairs += " " + remotePathWord(host) + " " + remotePathWord(inside);
            script += srunLine(id, "sirius-check", job.gpus) + " bash \"$LS\" --check" + pairs + " < /dev/null 2>&1\n";
            script += "echo \"@@check-exit $?\"\n";
            // a first exec of a large image on a network file system takes a while
            const ssh::CommandResult r = remote(script, std::chrono::seconds(300));
            if (keyValues(r.out)["script"] == "failed")
                throw Failure{Step::Checks, "The worker's launch script could not be written to ~/.sirius/run on " + p.host + ".", trim(r.err),
                              "Check that your home folder on the cluster is writable and not full."};
            const NodeCheckReport rep = readNodeChecks(r.out + (r.err.empty() ? std::string() : "\n" + r.err), job, buildJson, dir, buildInfo(), node, hint);
            update([&](Status& x) { x.nodeChecks = rep.checks; });
            if (const NodeCheck* f = rep.failure()) throw Failure{Step::Checks, f->detail, rep.output, f->fix, rep.noEngine};
            // one line of what was found, the worker's code first
            const auto find = [&rep](const std::string& name) -> const NodeCheck* {
                for (const NodeCheck& c : rep.checks)
                    if (c.name == name) return &c;
                return nullptr;
            };
            std::string detail = codeText;
            StepStatus mark = StepStatus::Done;
            if (const NodeCheck* c = find("launcher")) detail += " \xC2\xB7 " + c->detail;
            if (const NodeCheck* c = find("python")) detail += " \xC2\xB7 image: " + c->detail;
            if (const NodeCheck* c = find("torch")) detail += c->status == StepStatus::Done ? std::string(", torch") : std::string(" \xC2\xB7 no torch (models will not run)");
            const std::size_t binds = bindPairs(p.bind).size();
            if (binds > 0) detail += " \xC2\xB7 " + std::to_string(binds) + (binds == 1 ? " data folder" : " data folders");
            if (p.engine) {
                std::string note;
                {
                    const std::lock_guard<std::mutex> g(m);
                    note = status.engineBuildNote;
                }
                detail += dir.empty() ? " \xC2\xB7 " + note : " \xC2\xB7 engine " + shortCommit(dir) + " (" + note + ")";
            }
            for (const char* name : {"gpu", "cuda"})
                if (const NodeCheck* c = find(name)) {
                    detail += " \xC2\xB7 " + c->label + ": " + c->detail;
                    if (c->status == StepStatus::Warning) mark = StepStatus::Warning;
                }
            if (const NodeCheck* c = find("cache")) detail += " \xC2\xB7 node cache " + c->detail;
            if (const std::string w = emptyBindWarning(p); !w.empty()) {
                // not a failure: data in the home folder opens all the same
                mark = StepStatus::Warning;
                detail += " \xC2\xB7 " + w;
                say("Cluster: " + w);
            }
            stepState(Step::Checks, mark, detail);
            say("Cluster: the checks in job " + id + " on " + node + " passed");
        }

        // The worker as a step of the held job: app/python/slurm/
        // sirius_worker.sbatch (bash, not sbatch) under srun --overlap, in
        // the background and in a session of its own, so it outlives this
        // command and the SSH connection. The token goes to a 0600 file whose
        // name is all the step is given.
        void startStep(const Profile& p, const std::string& id) {
            stepState(Step::Start, StepStatus::Running, "srun --jobid=" + id);
            std::string dir, code;
            int gpus = 0;
            int seq = 0;
            {
                const std::lock_guard<std::mutex> g(m);
                dir = engineDir;
                code = workerDir;
                gpus = jobProfile.gpus;
                seq = ++workerSeq;
                token = ssh::randomHex(16);
            }
            const std::string logPath = "\"$HOME/.sirius/run/sirius-worker-" + id + "-" + std::to_string(seq) + ".log\"";
            std::string script = "umask 077\n";
            script += "mkdir -p \"$HOME/.sirius/run\" && chmod 700 \"$HOME/.sirius/run\" || exit 4\n";
            // token files of workers that never started, a day old
            script += "find \"$HOME/.sirius/run\" -maxdepth 1 -name 'token.*' -mmin +1440 -exec rm -f {} + 2>/dev/null\n";
            // this application's launch script, never the checkout's (which may be older)
            script += uploadLaunchScript();
            script += "cd \"$HOME\" || exit 3\n";
            // The token is written to a private file over this command
            // channel (printf is a builtin: no argument list shows it) and the
            // step is told only the file's name. The worker reads it and deletes it.
            script += "tf=$(mktemp \"$HOME/.sirius/run/token.XXXXXXXX\") || exit 4\n";
            script += "printf '%s' " + shellQuote(token) + " > \"$tf\" || { rm -f \"$tf\"; exit 4; }\n";
            script += "unset SIRIUS_TOKEN SIRIUS_VENV\nexport SIRIUS_TOKEN_FILE=\"$tf\"\n";
            // port 0: the worker takes a free one and says which in its log
            script += "export SIRIUS_PORT=0 SIRIUS_MAX_CLIENTS=8\n";
            script += stepEnvironment(p, dir, code, gpus);
            script += "S=; command -v setsid >/dev/null 2>&1 && S=setsid\n";
            const std::string srun = srunLine(id, "sirius-worker", gpus);
            script += "$S nohup " + srun + " bash \"$LS\" > " + logPath + " 2>&1 < /dev/null &\n";
            script += "echo \"srun=$!\"\n";
            const ssh::CommandResult r = remote(script, std::chrono::seconds(60));
            if (keyValues(r.out)["script"] == "failed")
                throw Failure{Step::Start, "The worker's launch script could not be written to ~/.sirius/run on " + p.host + ".", trim(r.err),
                              "Check that your home folder on the cluster is writable and not full."};
            if (!r.ok() || keyValues(r.out)["srun"].empty())
                throw Failure{Step::Start, "The worker could not be started in job " + id + ".", trim(r.err.empty() ? r.out : r.err), {}};
            {
                const std::lock_guard<std::mutex> g(m);
                workerLog = logPath;
                workerPort = 0;
            }
            say("Cluster: the worker starts in job " + id);
        }

        void waitForWorker(const Profile& p, const std::string& id, const std::string& node) {
            stepState(Step::Start, StepStatus::Running, "on " + node);
            const auto t0 = std::chrono::steady_clock::now();
            std::string logFile;
            {
                const std::lock_guard<std::mutex> g(m);
                logFile = workerLog;
            }
            // the step shows in squeue a moment after srun starts: only its
            // absence after that, or the job's end, means it is gone
            int missing = 0;
            for (;;) {
                const std::string script = "echo state=$(squeue -h -j " + id + " -o %T 2>/dev/null)\n" +
                                           "echo steps=$(squeue -h -s -j " + id + " -o '%j' 2>/dev/null | grep -c '^sirius-worker$')\n" + "[ -f " + logFile +
                                           " ] && grep -m1 -E '^\\{\"(port|error)\"' " + logFile + " | sed 's/^/announce=/'\ntrue\n";
                const ssh::CommandResult r = remote(script, std::chrono::seconds(30));
                auto kv = keyValues(r.out);
                const std::string announce = kv["announce"];
                if (!announce.empty()) {
                    json j;
                    try {
                        j = json::parse(announce);
                    } catch (const json::exception&) {
                    }
                    if (j.contains("error") && j["error"] == "engine_missing") {
                        const std::string at = j.value("path", std::string());
                        throw Failure{Step::Start,
                                      "No SIRIUS C++ engine found on " + node + (at.empty() ? std::string() : ": " + at + " is not there") +
                                          ", so the worker was not started (it never starts without the engine).",
                                      logTail(id), noEngineFix({}), true};
                    }
                    if (j.contains("error")) {
                        std::string missingPkgs;
                        if (j.contains("missing") && j["missing"].is_array())
                            for (const json& x : j["missing"]) missingPkgs += (missingPkgs.empty() ? "" : ", ") + x.get<std::string>();
                        throw Failure{Step::Start,
                                      "The worker on " + node + " cannot start: " + (missingPkgs.empty() ? std::string("a package is missing") : missingPkgs + " missing") + ".",
                                      logTail(id), "Build the worker image again with what is missing, or ask whoever builds it for the current one."};
                    }
                    if (j.contains("port") && j["port"].is_number_integer()) {
                        // the worker took a free port (--port 0) and says which
                        const int port = j["port"].get<int>();
                        if (port <= 0 || port > 65535)
                            throw Failure{Step::Start, "The worker on " + node + " announced port " + std::to_string(port) + ".", logTail(id), {}};
                        {
                            const std::lock_guard<std::mutex> g(m);
                            workerPort = port;
                        }
                        stepState(Step::Start, StepStatus::Done, "listening on " + node + ":" + std::to_string(port));
                        return;
                    }
                }
                if (trim(kv["state"]).empty() || (kv["state"] != "RUNNING" && kv["state"] != "COMPLETING")) {
                    const std::string fin = finalState(id);
                    throw Failure{Step::Start, "Job " + id + " ended before the worker listened: " + fin + ".", logTail(id), {}};
                }
                missing = trim(kv["steps"]) == "0" ? missing + 1 : 0;
                if (missing >= 3)
                    throw Failure{Step::Start, "The worker stopped before it listened (job " + id + " still holds " + node + ").", logTail(id),
                                  "Its log above says why; fix the image or the settings and press Start worker again."};
                stepState(Step::Start, StepStatus::Running, "on " + node + " \xC2\xB7 starting \xC2\xB7 " + elapsed(std::chrono::steady_clock::now() - t0));
                if (!sleepCancellable(queuePoll)) throw Failure{Step::Start, "Cancelled while the worker starts.", {}, {}};
            }
            (void)p;
        }

        void hello(const Profile& p, const std::string& node) {
            int port = 0;
            std::string tok;
            {
                const std::lock_guard<std::mutex> g(m);
                port = workerPort;
                tok = token;
            }
            stepState(Step::Hello, StepStatus::Running, node + ":" + std::to_string(port) + " through the SSH tunnel");
            int socks = 0;
            {
                auto s = sshSession();
                socks = s ? s->socksPort() : 0;
            }
            std::unique_ptr<RemoteWorker> w;
            try {
                w = RemoteWorker::connect(node, port, tok, std::chrono::seconds(20), [this] { return cancel.load(); }, socks);
            } catch (const CancelledError&) {
                throw Failure{Step::Hello, "Cancelled while the worker answers.", {}, {}};
            } catch (const std::exception& e) {
                const std::string what = e.what();
                if (what.find("protocol version mismatch") != std::string::npos)
                    throw Failure{Step::Hello, "The worker on " + node + " speaks another protocol than this application (" + buildInfo().build + ").", what,
                                  workerCodeFix(p)};
                throw Failure{Step::Hello, "Could not reach the worker on " + node + ":" + std::to_string(port) + " through the SSH tunnel.", what, {}};
            }
            w->setCancelGrace(std::chrono::milliseconds(0));
            const WorkerCapabilities caps = w->capabilities();
            // the Python worker of another version than this application's
            if (!caps.engine.is_object() && !caps.version.empty() && caps.version != buildInfo().version)
                throw Failure{Step::Hello,
                              "The worker on " + node + " is sirius_worker " + caps.version + ", this application is SIRIUS " + buildInfo().version + ".",
                              {},
                              workerCodeFix(p)};
            // SIRIUS's engine was asked for: a worker without it is not a
            // connection (built-in steps would be refused on every run)
            if (p.engine && !caps.engine.is_object())
                throw Failure{Step::Hello,
                              "The worker on " + node + " is the Python worker alone: there is no SIRIUS C++ engine in this job, and every built-in step needs it.",
                              {},
                              noEngineFix({}),
                              true};
            // SIRIUS's engine serves this application only when their
            // operations are the same (core/build_info.hpp): refused here, in
            // words, before anything is run on it.
            if (caps.engine.is_object())
                if (const std::string refusal = engineMismatch(buildInfo(), buildInfoFromJson(caps.engine)); !refusal.empty())
                    throw Failure{Step::Hello, refusal, {}, engineBuildFix(p.engineBuilds, buildInfo())};
            int kinds = 0;
            for (const std::string& meth : caps.methods)
                if (meth.rfind("run:", 0) == 0) ++kinds;
            {
                const std::lock_guard<std::mutex> g(controlMutex);
                control = std::move(w);
            }
            update([&](Status& x) {
                x.caps = caps;
                ++x.capsSerial;
            });
            StepStatus helloStatus = StepStatus::Done;
            std::string detail;
            if (caps.engine.is_object()) {
                const json python = caps.engine.value("python", json::object());
                detail = "SIRIUS engine " + caps.engine.value("build", caps.version) + " \xC2\xB7 " + caps.device + " \xC2\xB7 Python worker " +
                         python.value("state", std::string("disabled")) + " \xC2\xB7 session " + caps.engine.value("session", std::string());
            } else {
                detail = "sirius_worker " + caps.version + " \xC2\xB7 " + caps.device + " \xC2\xB7 " + std::to_string(kinds) + " step kinds";
            }
            // the job's GPU that the worker cannot compute on: named, with why
            if (const std::string note = unusableGpuNote(caps); !note.empty()) {
                helloStatus = StepStatus::Warning;
                detail += " \xC2\xB7 " + note;
            }
            stepState(Step::Hello, helloStatus, detail);
        }

        // Ends the job's worker steps (and closes the connection to it); the
        // job stays. Waits until squeue no longer lists them (20 s at most).
        void stopWorkerStep(const std::string& id) {
            {
                const std::lock_guard<std::mutex> g(controlMutex);
                control.reset();
            }
            if (id.empty()) return;
            try {
                for (int i = 0; i < 20; ++i) {
                    const std::vector<std::string> running = steps(id, "sirius-worker");
                    if (running.empty()) break;
                    if (i == 0) {
                        std::string cmd;
                        for (const std::string& s : running) cmd += "scancel " + shellQuote(s) + " 2>/dev/null\n";
                        remote(cmd + "true\n", std::chrono::seconds(30));
                    }
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                }
            } catch (const std::exception& e) {
                say(std::string("Cluster: the worker step could not be stopped: ") + e.what());
            }
            {
                const std::lock_guard<std::mutex> g(m);
                workerPort = 0;
            }
            update([&](Status& x) { x.caps = WorkerCapabilities{}; });
        }

        // After a reattach: the worker step this session started, when it
        // still runs and answers. False (and nothing changed) otherwise.
        bool reattachWorker(const Profile& p, const std::string& id, const std::string& node) {
            std::string logFile, tok;
            {
                const std::lock_guard<std::mutex> g(m);
                logFile = workerLog;
                tok = token;
            }
            if (logFile.empty() || tok.empty()) return false;
            try {
                if (steps(id, "sirius-worker").empty()) return false;
                stepState(Step::Checks, StepStatus::Done, "the worker of job " + id + " still runs: nothing to check");
                waitForWorker(p, id, node);
                hello(p, node);
                return true;
            } catch (const Failure& f) {
                say("Cluster: the worker of job " + id + " does not answer (" + f.reason + ")");
                for (const Step s : {Step::Checks, Step::Start, Step::Hello}) stepState(s, StepStatus::Pending, {});
                return false;
            }
        }

        // ================================================================================
        // The steps in order
        // ================================================================================

        // Step 1: logged in, the job held. Returns the node.
        std::string getJob(Profile& p) {
            auto s = sshSession();
            const bool reuse = s && s->isOpen() && s->host() == p.host;
            if (reuse) {
                stepState(Step::Login, StepStatus::Done, "logged in to " + p.host);
            } else {
                if (s) s->close();
                login(p);
            }
            listPartitions(p, false);
            fillDefaults(p);
            // A job of this session's that still runs (left running at a
            // disconnect, or the SSH connection dropped) is reattached to:
            // its worker may still hold the results computed there.
            std::string again, adopt;
            bool wasAdopted = false;
            {
                const std::lock_guard<std::mutex> g(m);
                again = reattachJob;
                reattachJob.clear();
                adopt = adoptId;
                adoptId.clear();
                wasAdopted = !again.empty() && again == adoptedJob;
            }
            if (!adopt.empty()) {
                takeUp(p, adopt);
            } else if (!again.empty() && jobRunning(again)) {
                update([&](Status& x) {
                    x.jobId = again;
                    x.adopted = wasAdopted;
                });
                stepState(Step::Submit, StepStatus::Done, "job " + again + " \xC2\xB7 reattached" + (wasAdopted ? std::string(" (yours)") : std::string()));
                say("Cluster: job " + again + " still runs: reattaching to it");
            } else {
                submitHolder(p);
            }
            std::string id;
            {
                const std::lock_guard<std::mutex> g(m);
                id = status.jobId;
            }
            return waitInQueue(p, id);
        }

        // What the profile leaves empty, from what the cluster said.
        void fillDefaults(Profile& p) {
            std::optional<ClusterInfo> ci;
            {
                const std::lock_guard<std::mutex> g(m);
                if (info && info->host == p.host) ci = info;
            }
            if (!ci) return;
            const std::vector<std::string> filled = fillFromCluster(p, *ci);
            if (filled.empty()) return;
            std::string line;
            for (const std::string& f : filled) line += (line.empty() ? "" : ", ") + f;
            say("Cluster: filled in from " + p.host + ": " + line);
            const std::lock_guard<std::mutex> g(m);
            profile = p;
        }

        // Step 2: the worker in the held job.
        void startWorkerIn(const Profile& p, const std::string& id, const std::string& node) {
            for (const Step s : {Step::Checks, Step::Start, Step::Hello}) stepState(s, StepStatus::Pending, {});
            update([&](Status& x) {
                x.state = State::Starting;
                x.since = std::chrono::steady_clock::now();
                x.reason.clear();
                x.remoteOutput.clear();
                x.fix.clear();
                x.noEngine = false;
                x.nodeChecks.clear();
            });
            stopWorkerStep(id);   // a new image, new data folders: the old worker goes first
            checks(p);
            checkInJob(p, id, node);
            startStep(p, id);
            waitForWorker(p, id, node);
            try {
                hello(p, node);
            } catch (const Failure& f) {
                // a worker that cannot serve this application holds no GPU meanwhile
                if (f.noEngine) stopWorkerStep(id);
                throw;
            }
        }

        void connected() {
            connecting.store(false);
            setState(State::Connected);
            Status st;
            {
                const std::lock_guard<std::mutex> g(m);
                st = status;
            }
            const std::string note = unusableGpuNote(st.caps);
            say("HPC: connected to the worker on " + st.node + " (job " + st.jobId + ", " + st.caps.device + (note.empty() ? std::string() : "; " + note) + ")");
            startKeeper();
        }

        void jobReady(const std::string& reason = {}) {
            connecting.store(false);
            update([&](Status& x) {
                x.state = State::JobReady;
                x.since = std::chrono::steady_clock::now();
                x.reason = reason;
                auto s2 = ssh;
                x.sshUp = s2 && s2->isOpen();
            });
            startKeeper();
        }

        void run() {
            Profile p;
            Mode how = Mode::Both;
            {
                const std::lock_guard<std::mutex> g(m);
                p = profile;
                how = mode;
            }
            bool jobHeld = how == Mode::Worker;
            try {
                std::string id, node;
                if (how == Mode::Login) {
                    auto s = sshSession();
                    if (!(s && s->isOpen() && s->host() == p.host)) {
                        if (s) s->close();
                        login(p);
                    } else {
                        stepState(Step::Login, StepStatus::Done, "logged in to " + p.host);
                    }
                    listPartitions(p, true);
                    fillDefaults(p);
                    // logged in, the partitions listed: nothing is submitted. The
                    // flag is cleared before the state is published (see below).
                    connecting.store(false);
                    setState(State::Idle);
                    return;
                }
                if (how != Mode::Worker) {
                    node = getJob(p);
                    jobHeld = true;
                    {
                        const std::lock_guard<std::mutex> g(m);
                        id = status.jobId;
                    }
                    // a reattached job's worker, when it still answers
                    if (reattachWorker(p, id, node)) {
                        connected();
                        return;
                    }
                    if (how == Mode::Job || how == Mode::Adopt) {
                        jobReady();
                        return;
                    }
                } else {
                    const std::lock_guard<std::mutex> g(m);
                    id = status.jobId;
                    node = status.node;
                }
                startWorkerIn(p, id, node);
                connected();
            } catch (const Failure& f) {
                // Cleared before the state is published: whoever sees the attempt
                // settle (Connect again, a test) must be able to start the next one,
                // which start() refuses while this flag is set.
                connecting.store(false);
                // the worker failed, the job holds on: JobReady, with why
                const bool keepJob = jobHeld && static_cast<int>(f.step) >= kJobStepCount && !cancelledJob();
                update([&](Status& x) {
                    x.steps[static_cast<std::size_t>(f.step)] = StepState{StepStatus::Failed, f.reason};
                    x.state = keepJob ? State::JobReady : State::Disconnected;
                    x.since = std::chrono::steady_clock::now();
                    x.reason = f.reason;
                    x.remoteOutput = f.remote;
                    if (!f.fix.empty()) x.fix = f.fix;
                    x.noEngine = f.noEngine;
                    auto s2 = ssh;
                    x.sshUp = s2 && s2->isOpen();
                });
                say("HPC: " + f.reason + (f.remote.empty() ? std::string() : " \xE2\x80\x94 " + f.remote.substr(0, 300)));
                if (keepJob) {
                    cancel.store(false);   // a Stop during the worker's start: the job is still watched
                    startKeeper();
                }
            } catch (const std::exception& e) {
                connecting.store(false);
                const bool keepJob = jobHeld && !cancelledJob();
                update([&](Status& x) {
                    x.state = keepJob ? State::JobReady : State::Disconnected;
                    x.since = std::chrono::steady_clock::now();
                    x.reason = cancel.load() ? std::string("Cancelled.") : std::string(e.what());
                    for (StepState& st : x.steps)
                        if (st.status == StepStatus::Running) st = StepState{StepStatus::Failed, x.reason};
                    auto s2 = ssh;
                    x.sshUp = s2 && s2->isOpen();
                });
                say(std::string("HPC: ") + (cancel.load() ? "cancelled" : e.what()));
                if (keepJob) {
                    cancel.store(false);
                    startKeeper();
                }
            }
            connecting.store(false);
        }

        // A cancel during the worker's start leaves the job held only when
        // the SSH connection is still there to watch it.
        bool cancelledJob() {
            auto s = sshSession();
            return !s || !s->isOpen();
        }

        // ================================================================================
        // Building an image
        // ================================================================================

        // Runs `body` (bash) as a step of the job, in the foreground: output and exit code.
        ssh::CommandResult inJob(const std::string& id, const std::string& body, std::chrono::milliseconds timeout) {
            auto s = sshSession();
            if (!s || !s->isOpen()) throw ssh::SshError("the SSH connection is closed");
            return s->run("srun --jobid=" + id + " --overlap --nodes=1 --ntasks=1 --job-name=sirius-probe bash -c " + shellQuote(body) + " < /dev/null 2>&1",
                          timeout, [this] { return buildCancel.load(); });
        }

        // Whether the cluster lets the user build an image: a tiny one built
        // with --fakeroot on the job's node (no download).
        bool probe(const Profile& p, const std::string& id, std::string& why) {
            std::string body = launcherScript(p);
            body += "[ -n \"$L\" ] || { echo 'neither apptainer nor singularity can be run here'; exit 9; }\n";
            body += "T=$(mktemp -d \"$HOME/.sirius/run/probe.XXXXXX\") || exit 9\n";
            body += "printf 'Bootstrap: scratch\\n%%files\\n    %s /probe.def\\n' \"$T/probe.def\" > \"$T/probe.def\"\n";
            body += "\"$L\" build --fakeroot \"$T/probe.sif\" \"$T/probe.def\" > \"$T/out\" 2>&1; rc=$?\n";
            body += "tail -n 5 \"$T/out\"; rm -rf \"$T\"; echo \"@@rc $rc\"\n";
            const ssh::CommandResult r = inJob(id, body, std::chrono::seconds(180));
            const std::string out = trim(r.out);
            if (out.find("@@rc 0") != std::string::npos) return true;
            std::string said = out;
            if (const std::size_t at = said.rfind("@@rc"); at != std::string::npos) said = trim(said.substr(0, at));
            why = said.empty() ? trim(r.err) : said;
            if (why.empty()) why = "the build of a test image failed";
            return false;
        }

        void build(const Profile& p, const std::string& id, const std::string& def, const std::string& image, bool probeOnly) {
            auto setBuild = [&](const std::function<void(BuildStatus&)>& fn) { update([&](Status& x) { fn(x.build); }); };
            try {
                setBuild([&](BuildStatus& b) {
                    b = BuildStatus{};
                    b.phase = BuildStatus::Phase::Probing;
                    b.image = image;
                    b.started = std::chrono::steady_clock::now();
                });
                std::string why;
                const bool ok = probe(p, id, why);
                setBuild([&](BuildStatus& b) {
                    b.supported = ok;
                    b.why = why;
                });
                if (!ok) {
                    setBuild([&](BuildStatus& b) {
                        b.phase = BuildStatus::Phase::Failed;
                        b.error = "This cluster does not let you build images (apptainer build --fakeroot was refused). Pick an image someone built "
                                  "under Worker image instead.";
                    });
                    say("Cluster: images cannot be built on " + p.host + ": " + why);
                    return;
                }
                if (probeOnly) {
                    setBuild([](BuildStatus& b) { b.phase = BuildStatus::Phase::None; });
                    return;
                }
                const std::string d = trim(def).empty() ? trim(p.checkout) + "/containers/sirius-worker.def" : trim(def);
                const ssh::CommandResult there = remote("[ -f " + remotePathWord(d) + " ] && echo def=yes || echo def=no\n[ -e " + remotePathWord(image) +
                                                            " ] && echo out=exists || echo out=new\n",
                                                        std::chrono::seconds(30));
                auto kv = keyValues(there.out);
                if (kv["def"] != "yes") {
                    setBuild([&](BuildStatus& b) {
                        b.phase = BuildStatus::Phase::Failed;
                        b.error = "There is no definition file at " + d + " on " + p.host +
                                  ": give the path of SIRIUS's worker definition (containers/sirius-worker.def in a SIRIUS checkout).";
                    });
                    return;
                }
                if (kv["out"] == "exists") {
                    setBuild([&](BuildStatus& b) {
                        b.phase = BuildStatus::Phase::Failed;
                        b.error = "There is a file at " + image + " already: choose another name for the new image (nothing is written over).";
                    });
                    return;
                }
                setBuild([](BuildStatus& b) { b.phase = BuildStatus::Phase::Building; });
                say("Cluster: building " + image + " in job " + id);
                int seq = 0;
                {
                    const std::lock_guard<std::mutex> g(m);
                    seq = ++workerSeq;
                }
                const std::string logPath = "\"$HOME/.sirius/run/sirius-build-" + id + "-" + std::to_string(seq) + ".log\"";
                std::string body = launcherScript(p);
                body += "[ -n \"$L\" ] || { echo 'neither apptainer nor singularity can be run here'; echo '@@rc 9'; exit 9; }\n";
                body += "\"$L\" build --fakeroot " + remotePathWord(image) + " " + remotePathWord(d) + " 2>&1; echo \"@@rc $?\"\n";
                std::string script = "umask 022\nS=; command -v setsid >/dev/null 2>&1 && S=setsid\n";
                script += "$S nohup srun --jobid=" + id + " --overlap --nodes=1 --ntasks=1 --job-name=sirius-build bash -c " + shellQuote(body) + " > " +
                          logPath + " 2>&1 < /dev/null &\necho started=1\n";
                if (keyValues(remote(script, std::chrono::seconds(60)).out)["started"] != "1") throw std::runtime_error("the build could not be started");
                for (;;) {
                    if (buildCancel.load()) {
                        for (const std::string& s : steps(id, "sirius-build")) remote("scancel " + shellQuote(s) + " 2>/dev/null; true", std::chrono::seconds(30));
                        setBuild([](BuildStatus& b) {
                            b.phase = BuildStatus::Phase::Failed;
                            b.error = "The build was stopped.";
                        });
                        return;
                    }
                    const std::string tail = remote("tail -n 15 " + logPath + " 2>/dev/null; true", std::chrono::seconds(30)).out;
                    std::string shown = tail;
                    const std::size_t rc = tail.find("@@rc ");
                    if (rc != std::string::npos) shown = tail.substr(0, rc);
                    setBuild([&](BuildStatus& b) { b.log = trim(shown); });
                    if (rc != std::string::npos) {
                        const bool ok2 = std::atoi(tail.c_str() + rc + 5) == 0;
                        setBuild([&](BuildStatus& b) {
                            b.phase = ok2 ? BuildStatus::Phase::Done : BuildStatus::Phase::Failed;
                            if (!ok2) b.error = "The build failed: its last lines are below.";
                        });
                        say(ok2 ? "Cluster: built " + image : "Cluster: the build of " + image + " failed");
                        return;
                    }
                    if (!jobRunning(id)) {
                        setBuild([&](BuildStatus& b) {
                            b.phase = BuildStatus::Phase::Failed;
                            b.error = "Job " + id + " ended during the build.";
                        });
                        return;
                    }
                    for (int i = 0; i < 20 && !buildCancel.load(); ++i) std::this_thread::sleep_for(queuePoll / 20);
                }
            } catch (const std::exception& e) {
                setBuild([&](BuildStatus& b) {
                    b.phase = BuildStatus::Phase::Failed;
                    b.error = std::string("The build could not go on: ") + e.what();
                });
            }
        }

        // --- keep-alive ------------------------------------------------------------------
        void startKeeper() {
            stopKeeper.store(false);
            if (keeper.joinable() && keeper.get_id() != std::this_thread::get_id()) keeper.join();
            keeper = std::thread([this] { keep(); });
        }

        // The job or the SSH connection went: disconnected.
        void lost(const std::string& reason, const std::string& remoteText = {}, bool jobEnded = false) {
            {
                const std::lock_guard<std::mutex> g(controlMutex);
                control.reset();
            }
            update([&](Status& x) {
                x.state = State::Disconnected;
                x.since = std::chrono::steady_clock::now();
                x.reason = reason;
                x.remoteOutput = remoteText;
                x.jobEnded = jobEnded;
                x.dropped = true;
                auto s = ssh;
                x.sshUp = s && s->isOpen();
            });
            say("HPC: disconnected: " + reason);
        }

        // The worker went, the job holds on: JobReady, with why.
        void workerGone(const std::string& reason, const std::string& remoteText) {
            {
                const std::lock_guard<std::mutex> g(controlMutex);
                control.reset();
            }
            update([&](Status& x) {
                x.state = State::JobReady;
                x.since = std::chrono::steady_clock::now();
                x.reason = reason;
                x.remoteOutput = remoteText;
                x.caps = WorkerCapabilities{};
                x.steps[static_cast<std::size_t>(Step::Hello)] = StepState{StepStatus::Failed, reason};
            });
            say("HPC: " + reason + "; job still held");
        }

        // SIRIUS's engine answered its hello before its Python child was up
        // (torch takes seconds to minutes to import): that child is still to
        // come, and the engine can say again ("capabilities").
        static bool pythonPending(const WorkerCapabilities& caps) {
            if (!caps.engine.is_object() || std::find(caps.methods.begin(), caps.methods.end(), "capabilities") == caps.methods.end()) return false;
            const json python = caps.engine.value("python", json::object());
            const std::string state = python.is_object() ? python.value("state", std::string()) : std::string();
            return state == "starting" || state == "idle";
        }

        // The engine's capabilities asked again: the Python child's state,
        // torch, the methods it adds (model steps, plugins). Status::caps
        // and capsSerial change when they do.
        void refreshed(const WorkerCapabilities& fresh) {
            std::string before, after, error, torch;
            bool differs = false;
            update([&](Status& x) {
                const auto state = [](const WorkerCapabilities& c) {
                    const json py = c.engine.is_object() ? c.engine.value("python", json::object()) : json::object();
                    return py.is_object() ? py.value("state", std::string()) : std::string();
                };
                before = state(x.caps);
                WorkerCapabilities c = fresh;
                c.protocolVersion = x.caps.protocolVersion;
                if (!c.engine.is_object()) return;   // not the engine's answer: keep what the hello said
                after = state(c);
                differs = after != before || c.methods != x.caps.methods || c.torch != x.caps.torch;
                if (!differs) return;
                const json py = c.engine.value("python", json::object());
                if (py.is_object()) error = py.value("error", std::string());
                torch = c.torch;
                x.caps = std::move(c);
                ++x.capsSerial;
            });
            if (!differs || after == before) return;
            if (after == "ready") say("HPC: the Python worker beside the engine is ready" + (torch.empty() ? std::string(" (no torch)") : " (torch " + torch + ")"));
            else if (after == "failed") say("HPC: the Python worker beside the engine did not start: " + error);
        }

        void keep() {
            auto lastJobCheck = std::chrono::steady_clock::now();
            for (;;) {
                bool pending = false;
                {
                    const std::lock_guard<std::mutex> g(m);
                    pending = status.state == State::Connected && pythonPending(status.caps);
                }
                // more often while the engine's Python child comes up
                const std::chrono::milliseconds wait = pending ? std::min(keepAlive, std::chrono::milliseconds(3000)) : keepAlive;
                {
                    std::unique_lock<std::mutex> lk(keeperMutex);
                    if (keeperWake.wait_for(lk, wait, [this] { return stopKeeper.load(); })) return;
                }
                std::string id, host;
                State state = State::Idle;
                {
                    const std::lock_guard<std::mutex> g(m);
                    id = status.jobId;
                    host = status.host;
                    state = status.state;
                }
                if (state != State::Connected && state != State::JobReady) return;
                auto s = sshSession();
                if (!s || !s->isOpen()) {
                    lost("the SSH connection to " + host + " ended", s ? s->stderrTail() : std::string());
                    return;
                }
                bool pingFailed = false;
                std::string why;
                std::optional<WorkerCapabilities> fresh;
                {
                    const std::lock_guard<std::mutex> g(controlMutex);
                    if (control) {
                        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
                        const auto stop = [&] { return stopKeeper.load() || std::chrono::steady_clock::now() > deadline; };
                        try {
                            control->call("ping", json::object(), {}, {}, stop);
                        } catch (const std::exception& e) {
                            if (stopKeeper.load()) return;
                            pingFailed = true;
                            why = isCancellation(e) ? std::string("no answer within 20 s") : std::string(e.what());
                        }
                        if (!pingFailed && pending) {
                            try {
                                fresh = parseWorkerCapabilities(control->call("capabilities", json::object(), {}, {}, stop).result);
                            } catch (const std::exception& e) {
                                if (stopKeeper.load()) return;
                                say(std::string("HPC: the engine's capabilities could not be asked again: ") + e.what());
                            }
                        }
                    }
                }
                if (fresh) refreshed(*fresh);
                const auto now = std::chrono::steady_clock::now();
                if (pingFailed || now - lastJobCheck >= 2 * keepAlive) {
                    lastJobCheck = now;
                    std::string jobState;
                    try {
                        jobState = trim(remote("squeue -h -j " + id + " -o %T 2>/dev/null", std::chrono::seconds(30)).out);
                    } catch (const std::exception&) {
                        jobState = "?";
                    }
                    if (stopKeeper.load()) return;
                    if (jobState.empty() || (jobState != "RUNNING" && jobState != "COMPLETING" && jobState != "?")) {
                        lost("job " + id + " ended: " + finalState(id), logTail(id), true);
                        return;
                    }
                    if (pingFailed) {
                        workerGone("the worker stopped answering (" + why + ")", logTail(id));
                        continue;
                    }
                }
            }
        }

        void stopKeeperThread() {
            stopKeeper.store(true);
            keeperWake.notify_all();
            if (keeper.joinable() && keeper.get_id() != std::this_thread::get_id()) keeper.join();
        }

        void stopWorkerThread() {
            cancel.store(true);
            abortLogin.store(true);
            if (worker.joinable()) worker.join();
            cancel.store(false);
        }

        void stopBuilderThread() {
            buildCancel.store(true);
            if (builder.joinable()) builder.join();
            buildCancel.store(false);
        }
    };

    ConnectionBadge noEngineBadge(ConnectionBadge b, const Status& st, const std::string& why) {
        if (why.empty()) return b;
        if (b.kind == ConnectionBadge::Kind::Off || b.kind == ConnectionBadge::Kind::JobReady ||
            (b.kind == ConnectionBadge::Kind::Connected && !hasEngine(st.caps))) {
            b.label = (b.kind == ConnectionBadge::Kind::Off || st.node.empty() ? std::string("Cluster") : shortNodeName(st.node)) + " \xC2\xB7 no engine";
            b.kind = ConnectionBadge::Kind::Failed;
            b.tooltip = why + "\n" + b.tooltip;
        }
        return b;
    }

    // --- the user's own jobs -----------------------------------------------------------

    std::string userJobsScript() {
        // every job of the user's that runs or waits, SIRIUS's or not
        return "u=\"${USER:-$(id -un)}\"\n"
               "squeue -h -u \"$u\" -t RUNNING,PENDING -o '%i|%j|%P|%T|%M|%l|%D|%R|%b|%C|%m' 2>&1\n";
    }

    int gresGpus(const std::string& gres) {
        // "gpu:2", "gres:gpu:a100:2", "gres/gpu:a100=2", "gres/gpu=2", "gres/gpu" (one); several entries, comma separated
        int total = 0;
        for (const std::string& raw : splitOn(gres, ',')) {
            const std::string e = stripped(raw);
            const std::size_t at = e.find("gpu");
            if (at == std::string::npos) continue;
            std::string rest = e.substr(at + 3);
            // the count: the last number after ':' or '=' (a type name such as "a100" is not one)
            int n = 1;
            const std::size_t sep = rest.find_last_of(":=");
            if (sep != std::string::npos) {
                const std::string tail = rest.substr(sep + 1);
                std::size_t digits = 0;
                while (digits < tail.size() && std::isdigit(static_cast<unsigned char>(tail[digits]))) ++digits;
                if (digits > 0) n = std::atoi(tail.substr(0, digits).c_str());
            }
            total += n;
        }
        return total;
    }

    std::vector<ClusterJob> parseUserJobs(const std::string& output) {
        std::vector<ClusterJob> out;
        std::istringstream in(output);
        std::string line;
        while (std::getline(in, line)) {
            const std::vector<std::string> f = splitOn(stripped(line), '|');
            if (f.size() < 8 || stripped(f[0]).empty() || !std::isdigit(static_cast<unsigned char>(stripped(f[0])[0]))) continue;
            ClusterJob j;
            j.id = stripped(f[0]);
            j.name = stripped(f[1]);
            j.partition = stripped(f[2]);
            j.state = stripped(f[3]);
            j.used = stripped(f[4]);
            j.limit = stripped(f[5]);
            j.nodes = std::atoi(stripped(f[6]).c_str());
            j.where = stripped(f[7]);
            if (f.size() > 8) j.gpus = gresGpus(stripped(f[8]));
            if (f.size() > 9) j.cpus = std::atoi(stripped(f[9]).c_str());
            if (f.size() > 10) {
                const std::string mem = stripped(f[10]);
                if (mem != "0" && mem != "N/A") j.mem = mem;
            }
            out.push_back(j);
        }
        return out;
    }

    std::string jobSummary(const ClusterJob& job) {
        std::vector<std::string> parts;
        if (job.running()) {
            if (!job.where.empty()) parts.push_back(job.where);
        } else {
            std::string why = job.where;
            if (why.size() > 1 && why.front() == '(' && why.back() == ')') why = why.substr(1, why.size() - 2);
            parts.push_back(lowerCase(job.state) + (why.empty() || why == "None" ? std::string() : " (" + why + ")"));
        }
        if (job.gpus > 0) parts.push_back(std::to_string(job.gpus) + (job.gpus == 1 ? " GPU" : " GPUs"));
        else parts.push_back("no GPU");
        if (job.cpus > 0) parts.push_back(std::to_string(job.cpus) + (job.cpus == 1 ? " CPU" : " CPUs"));
        if (!job.mem.empty()) parts.push_back(job.mem);
        if (job.running() && !job.used.empty()) parts.push_back(job.used + (job.limit.empty() ? std::string() : " of " + job.limit));
        else if (!job.limit.empty()) parts.push_back("time limit " + job.limit);
        std::string s;
        for (const std::string& part : parts) s += (s.empty() ? "" : " \xC2\xB7 ") + part;
        return s;
    }

    // --- the worker's launch script -------------------------------------------------

    const std::string& workerLaunchScript() {
        static const std::string text(reinterpret_cast<const char*>(kSiriusWorkerScript), static_cast<std::size_t>(kSiriusWorkerScript_size));
        return text;
    }

    std::string workerLaunchScriptName() {
        std::string build = buildInfo().build;
        for (char& c : build)
            if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '.' || c == '-' || c == '+' || c == '_')) c = '_';
        return "sirius_worker-" + (build.empty() ? std::string("unknown") : build) + ".sbatch";
    }

    namespace {
        unsigned byte(char c) { return static_cast<unsigned>(static_cast<unsigned char>(c)); }

        std::string base64(const std::string& in) {
            static const char* const k = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            std::string out;
            out.reserve((in.size() + 2) / 3 * 4);
            std::size_t i = 0;
            for (; i + 2 < in.size(); i += 3) {
                const unsigned v = (byte(in[i]) << 16) | (byte(in[i + 1]) << 8) | byte(in[i + 2]);
                out += k[(v >> 18) & 63];
                out += k[(v >> 12) & 63];
                out += k[(v >> 6) & 63];
                out += k[v & 63];
            }
            if (i < in.size()) {
                const bool two = i + 1 < in.size();
                const unsigned v = (byte(in[i]) << 16) | (two ? byte(in[i + 1]) << 8 : 0u);
                out += k[(v >> 18) & 63];
                out += k[(v >> 12) & 63];
                out += two ? k[(v >> 6) & 63] : '=';
                out += '=';
            }
            return out;
        }
    } // namespace

    std::string uploadLaunchScript() {
        // base64 on the command channel: the script's quotes and parentheses
        // never meet the shell that writes it
        return "LS=\"$HOME/.sirius/run/" + workerLaunchScriptName() +
               "\"\n"
               "{ printf '%s' '" +
               base64(workerLaunchScript()) +
               "' | base64 -d > \"$LS.part\" && chmod 700 \"$LS.part\" && mv -f \"$LS.part\" \"$LS\"; } || { rm -f \"$LS.part\"; echo script=failed; exit 6; }\n";
    }

    std::string engineBuildFix(const std::string& folder, const BuildInfo& app) {
        const std::string where = trim(folder).empty() ? std::string("your engine builds folder") : trim(folder);
        const std::string commit = app.commit.empty() || app.commit == "unknown" ? std::string("this application's commit") : app.commit;
        return "Build the engine of this SIRIUS into " + where + "/" + commit +
               " on the cluster: latents' scripts/build_sirius_engine.sbatch does it for a commit (app/python/slurm/README.md, \"Engine builds\", has "
               "the cmake line). Then Start worker again. A build of another commit serves this application only when its operations and engine API "
               "are the same.";
    }

    std::string noEngineFix(const std::string& hint) {
        return "Set Engine builds folder (Job \xE2\x96\xB8 More options) to the folder holding per-commit builds, e.g. " +
               (trim(hint).empty() ? std::string("~/sirius-builds") : trim(hint)) + ", then Start worker again.";
    }

    // --- the engine builds ----------------------------------------------------------

    std::string engineBuildsScript(const std::string& folder) {
        // The newest first; each line: "@@build <dir> <bin yes|nox|no> <python yes|no> <lib yes|no> <BUILD.json on one line>".
        // bin/sirius-cli is judged by its mode (ls -lL: a file, or a link to one, with an x in its
        // mode string), never by [ -x ]: that asks access(X_OK), which a login node answers "no"
        // for every file of a scratch it mounts noexec (fiona's /clusterfs/nvme2) though the
        // compute nodes run it. Whether it runs is checked in the job.
        return "EB=" + remotePathWord(folder) +
               "\n"
               "if [ -d \"$EB\" ]; then echo builds=yes\n"
               "  for d in $(ls -1t \"$EB\" 2>/dev/null); do\n"
               "    [ -f \"$EB/$d/BUILD.json\" ] || continue\n"
               "    x=no; f=\"$EB/$d/bin/sirius-cli\"\n"
               "    if [ -f \"$f\" ]; then x=nox; case \"$(ls -lLd \"$f\" 2>/dev/null)\" in -??[xs]* | -?????[xs]* | -????????[xt]*) x=yes ;; esac; fi\n"
               "    y=no; [ -f \"$EB/$d/python/sirius_worker/__main__.py\" ] && y=yes\n"
               "    l=no; [ -d \"$EB/$d/lib\" ] && l=yes\n"
               "    printf '@@build %s %s %s %s ' \"$d\" \"$x\" \"$y\" \"$l\"; tr -d '\\n\\r' < \"$EB/$d/BUILD.json\"; echo\n"
               "  done\n"
               "else echo builds=no; fi\n";
    }

    std::vector<EngineBuild> parseEngineBuilds(const std::string& output) {
        std::vector<EngineBuild> out;
        std::istringstream in(output);
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.rfind("@@build ", 0) != 0) continue;
            std::string rest = line.substr(8);
            const auto word = [&rest]() {
                const std::size_t a = rest.find_first_not_of(' ');
                if (a == std::string::npos) return std::string();
                const std::size_t b = rest.find(' ', a);
                return rest.substr(a, b == std::string::npos ? std::string::npos : b - a);
            };
            const auto drop = [&rest]() {
                const std::size_t a = rest.find_first_not_of(' ');
                const std::size_t b = a == std::string::npos ? std::string::npos : rest.find(' ', a);
                rest = b == std::string::npos ? std::string() : rest.substr(b);
            };
            EngineBuild b;
            b.dir = word();
            drop();
            // the columns: bin, then python and lib (a listing of before has fewer: the JSON follows)
            std::vector<std::string> cols;
            for (std::string w = word(); cols.size() < 3 && (w == "yes" || w == "no" || w == "nox"); w = word()) {
                cols.push_back(w);
                drop();
            }
            if (!cols.empty()) {
                b.runnable = cols[0] == "yes";
                b.binThere = cols[0] == "yes" || cols[0] == "nox";
            }
            if (cols.size() > 1) b.python = cols[1] == "yes";
            if (cols.size() > 2) b.lib = cols[2] == "yes";
            const json j = json::parse(rest, nullptr, false);
            if (j.is_object()) {
                b.readable = true;
                b.info = buildInfoFromJson(j);
            }
            if (!b.dir.empty()) out.push_back(b);
        }
        return out;
    }

    int pickEngineBuild(const std::vector<EngineBuild>& builds, const BuildInfo& app, std::string* note) {
        const auto whole = [](const EngineBuild& b) { return b.runnable && b.readable && b.lib; };
        // this application's own commit first
        for (std::size_t i = 0; i < builds.size(); ++i) {
            const EngineBuild& b = builds[i];
            if (whole(b) && !app.commit.empty() && app.commit != "unknown" && b.info.commit == app.commit && engineMismatch(app, b.info).empty()) {
                if (note) *note = "this build";
                return static_cast<int>(i);
            }
        }
        // else the newest with the same operations and engine API
        for (std::size_t i = 0; i < builds.size(); ++i) {
            const EngineBuild& b = builds[i];
            if (whole(b) && engineMismatch(app, b.info).empty()) {
                if (note) *note = "the same operations as this build";
                return static_cast<int>(i);
            }
        }
        if (note) {
            // Say why each build was passed over: a build whose operations match
            // but whose folder is not whole is a broken install, not a different
            // version, and the fix is a different one.
            std::size_t broken = 0, unreadable = 0, mismatched = 0;
            std::string firstBroken, what;
            for (const EngineBuild& b : builds) {
                if (!b.readable) ++unreadable;
                else if (!engineMismatch(app, b.info).empty()) ++mismatched;
                else if (!whole(b)) {
                    ++broken;
                    if (firstBroken.empty()) {
                        firstBroken = b.dir.substr(0, 12);
                        what = !b.binThere ? "has no bin/sirius-cli"
                                           : (!b.runnable ? "has a bin/sirius-cli without an execute bit in its mode (chmod +x bin/sirius-cli)"
                                                          : "has no lib/ folder");
                    }
                }
            }
            if (builds.empty()) *note = "The engine builds folder holds no build (<commit>/BUILD.json and bin/sirius-cli).";
            else if (broken > 0)
                *note = "Engine build " + firstBroken + " matches this application but " + what + ": reinstall that build" +
                        (broken + unreadable + mismatched > 1 ? " (" + std::to_string(builds.size()) + " builds checked)." : std::string("."));
            else if (mismatched == builds.size())
                *note = "None of the " + std::to_string(builds.size()) + (builds.size() == 1 ? " engine build" : " engine builds") +
                        " fits this application (" + app.build + "): their operations or engine API differ.";
            else
                *note = "None of the " + std::to_string(builds.size()) + (builds.size() == 1 ? " engine build" : " engine builds") +
                        " can be used: " + std::to_string(unreadable) + " with an unreadable BUILD.json, " + std::to_string(mismatched) +
                        " for other operations or engine API.";
        }
        return -1;
    }

    // --- the checks in the job ----------------------------------------------------------

    const NodeCheck* NodeCheckReport::failure() const {
        for (const NodeCheck& c : checks)
            if (c.status == StepStatus::Failed) return &c;
        return nullptr;
    }

    NodeCheckReport readNodeChecks(const std::string& output, const Profile& p, const std::optional<BuildInfo>& buildJson, const std::string& buildDir,
                                   const BuildInfo& app, const std::string& node, const std::string& buildsHint) {
        NodeCheckReport rep;
        const std::string on = node.empty() ? std::string("the job's node") : node;
        const std::string image = trim(p.container);
        const std::string launcher = p.launcher.empty() ? std::string("apptainer") : p.launcher;
        const std::string ask = "Build the worker image again (\"Build an image\", or app/python/slurm/README.md), or ask whoever builds it on this "
                                "cluster for the current one, and pick it under Worker image.";
        const std::vector<std::string> binds = bindHostPaths(p.bind);
        std::optional<json> version;
        std::vector<std::string> raw;
        bool imageStarted = false, anyCheck = false;
        const auto add = [&rep](std::string name, std::string label, StepStatus st, std::string detail, std::string fix = {}) {
            rep.checks.push_back(NodeCheck{std::move(name), std::move(label), st, std::move(detail), std::move(fix)});
        };
        std::istringstream in(output);
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.rfind("@@check-exit ", 0) == 0) {
                rep.exitCode = std::atoi(line.c_str() + 13);
                continue;
            }
            if (line.rfind("@@version ", 0) == 0) {
                const json j = json::parse(line.substr(10), nullptr, false);
                if (!j.is_discarded()) version = j;
                continue;
            }
            if (line.rfind("check ", 0) != 0) {
                if (!trim(line).empty()) raw.push_back(line);
                continue;
            }
            std::istringstream words(line.substr(6));
            std::string name, word, rest;
            words >> name >> word;
            std::getline(words, rest);
            rest = trim(rest);
            const StepStatus st = word == "ok" ? StepStatus::Done : (word == "warn" ? StepStatus::Warning : StepStatus::Failed);
            const bool ok = st == StepStatus::Done;
            anyCheck = true;
            if (name == "launcher") {
                if (ok) add(name, "Launcher", st, rest);
                else
                    add(name, "Launcher", st, "Neither " + launcher + " nor singularity can be run on " + on + " (module load was tried).",
                        "Ask your cluster's support how to run apptainer on its compute nodes, or set \"Container launcher\" under Job \xE2\x96\xB8 More "
                        "options to its full path there.");
            } else if (name == "image") {
                if (ok) {
                    imageStarted = true;
                    add(name, "Image", st, image + " starts on " + on);
                } else if (rest.find("not readable") != std::string::npos) {
                    add(name, "Image", st, "The worker image " + image + " cannot be read on " + on + ".",
                        "Make it readable to you (chmod a+r), or pick a copy you can read under Worker image.");
                } else {
                    add(name, "Image", st, "The worker image " + image + " cannot run the worker on " + on + ": " + rest + ".", ask);
                }
            } else if (name == "python") {
                if (ok) add(name, "Python", st, rest);
                else add(name, "Python", st, "The worker image " + image + " cannot run the worker: sirius and numpy do not import in it.", ask);
            } else if (name == "torch") {
                add(name, "torch", st, rest, ok ? std::string() : std::string("Build the image with torch (app/python/slurm/README.md) for the model steps."));
            } else if (name == "worker") {
                if (ok) add(name, "Worker's code", st, rest);
                else
                    add(name, "Worker's code", st, "The worker's code is not there on " + on + ": " + rest + ".",
                        "Check that the SIRIUS checkout (or the engine build) is on a file system the compute nodes see, readable to you.");
            } else if (name == "engine" || name == "engine-missing") {
                if (ok) {
                    // judged below, from what `version` said
                    add("engine", "C++ engine", st, rest);
                    continue;
                }
                rep.noEngine = true;
                if (name == "engine-missing") {
                    if (!buildDir.empty())
                        add("engine", "C++ engine", st,
                            "No SIRIUS C++ engine found: engine build " + buildDir + "'s bin/sirius-cli cannot be run on " + on + " (" + rest + ").",
                            "The build's folder must be on a file system the compute nodes run programs from (scratch usually is: ask your cluster's "
                            "support otherwise), and bin/sirius-cli an executable of this node's architecture; reinstall the build if it is not.");
                    else if (!trim(p.engineBin).empty())
                        add("engine", "C++ engine", st, "No SIRIUS C++ engine found: " + trim(p.engineBin) + " is neither an executable on " + on + " nor in the image.",
                            noEngineFix(buildsHint));
                    else
                        add("engine", "C++ engine", st,
                            "No SIRIUS C++ engine found: the image " + image + " has no /opt/sirius/bin/sirius-cli, and no Engine builds folder is set.",
                            noEngineFix(buildsHint));
                } else {
                    add("engine", "C++ engine", st, "SIRIUS's engine does not run in the image on " + on + ": `" + rest + "` (its output is under Details).",
                        "Its output says why: usually a library the build needs that is neither in the image nor in the build's lib/ folder (build the "
                        "engine against this image: app/python/slurm/README.md, \"Engine builds\"), or, with \"Permission denied\", a folder the "
                        "compute nodes mount noexec. Then Start worker again.");
                }
            } else if (name == "gpu") {
                std::string fix;
                if (!ok)
                    fix = rest.find("named no GPU") != std::string::npos
                              ? "The engine may compute on GPUs other jobs hold: ask your cluster's support how a job's GPUs are confined there "
                                "(CUDA_VISIBLE_DEVICES), or set GPUs to 0 on the Job page."
                              : "Check the job's GPUs on the Job page, and that the image is entered with the NVIDIA driver (--nv) on a GPU partition.";
                add(name, "GPUs", st, rest, fix);
            } else if (name.rfind("data:", 0) == 0) {
                const std::size_t i = static_cast<std::size_t>(std::atoi(name.c_str() + 5));
                const std::string folder = i < binds.size() ? binds[i] : rest;
                if (ok) add(name, "Data folder", st, folder);
                else if (rest.find("is not on") != std::string::npos)
                    add(name, "Data folder", st, "The data folder " + folder + " is not a folder on " + on + ": apptainer would stop before the worker starts.",
                        "Correct it or remove it under Data folders (the folders on the cluster your datasets live in).");
                else
                    add(name, "Data folder", st, "The data folder " + folder + " cannot be read inside the image on " + on + ".",
                        "Make it readable to you, or correct it under Data folders.");
            } else if (name == "cache") {
                if (ok) add(name, "Node cache", st, trim(p.scratch).empty() ? rest : trim(p.scratch));
                else
                    add(name, "Node cache", st, "You cannot write to the node cache folder " + trim(p.scratch) + " on " + on + ".",
                        "Choose another folder under Job \xE2\x96\xB8 More options, or leave it empty to use the node's temporary folder.");
            } else {
                add(name, name, st, rest);
            }
        }
        // the engine's own answer: SIRIUS's engine, the build BUILD.json says, one that serves this application
        for (NodeCheck& c : rep.checks) {
            if (c.name != "engine" || c.status != StepStatus::Done) continue;
            // sirius-cli's envelope ({"command": "version", "ok": true, "result": {...}}), or the result alone
            if (version && version->is_object() && version->contains("result") && (*version)["result"].is_object()) version = json((*version)["result"]);
            const json* build = version && version->is_object() && version->contains("build") && (*version)["build"].is_object() ? &(*version)["build"] : nullptr;
            if (!build) {
                c.status = StepStatus::Failed;
                c.detail = "The engine " + c.detail + " did not answer `version` as SIRIUS's engine does, on " + on + ".";
                c.fix = trim(p.engineBin).empty() ? engineBuildFix(p.engineBuilds, app) : noEngineFix(buildsHint);
                rep.noEngine = true;
                break;
            }
            const BuildInfo v = buildInfoFromJson(*build);
            const auto known = [](const std::string& commit) { return !commit.empty() && commit != "unknown"; };
            if (buildJson && ((known(v.commit) && known(buildJson->commit) && v.commit != buildJson->commit) || v.opsSchema != buildJson->opsSchema ||
                              v.api != buildJson->api)) {
                c.status = StepStatus::Failed;
                c.detail = "Engine build " + buildDir + "'s bin/sirius-cli is " + (v.build.empty() ? v.commit : v.build) + ", but its BUILD.json says " +
                           (buildJson->build.empty() ? buildJson->commit : buildJson->build) + ": the folder holds another build's executable.";
                c.fix = "Reinstall that build (bin/sirius-cli, lib/, python/ and BUILD.json of one commit), or remove it from the builds folder; then Start worker again.";
                rep.noEngine = true;
                break;
            }
            if (const std::string refusal = engineMismatch(app, v); !refusal.empty()) {
                c.status = StepStatus::Failed;
                c.detail = refusal;
                c.fix = engineBuildFix(p.engineBuilds, app);
                break;
            }
            std::string what = v.build.empty() ? v.version : v.build;
            if (known(v.commit)) what += " (" + v.commit.substr(0, 10) + ")";
            c.detail = what + " runs in the image" + (buildDir.empty() ? std::string() : ", from " + buildDir);
            // its CUDA, when the job has GPUs
            const json features = version->value("features", json::object());
            if (p.gpus > 0 && features.is_object()) {
                const bool cuda = features.value("cuda", false);
                const int devices = features.contains("cuda_devices") && features["cuda_devices"].is_number_integer() ? features["cuda_devices"].get<int>() : 0;
                if (!cuda)
                    rep.checks.push_back(NodeCheck{"cuda", "CUDA", StepStatus::Warning, "this engine was built without CUDA: every step runs on the CPU",
                                                   "Use an engine build made with CUDA (app/python/slurm/README.md, \"Engine builds\")."});
                else if (devices <= 0)
                    rep.checks.push_back(NodeCheck{"cuda", "CUDA", StepStatus::Warning, "the engine's CUDA finds no GPU in the step",
                                                   "Check that the image is entered with the NVIDIA driver (--nv) and the job holds a GPU."});
                else
                    rep.checks.push_back(NodeCheck{"cuda", "CUDA", StepStatus::Done, std::to_string(devices) + (devices == 1 ? " device" : " devices") + " for the engine", {}});
            }
            break;
        }
        // nothing at all came back: the step did not run; else an image that did not start
        if (!anyCheck) {
            add("step", "Check step", StepStatus::Failed, "The checks could not run in the job on " + on + ".",
                "The output under Details says why (srun's own words); press Start worker again, or Change job if the job is gone.");
        } else if (!imageStarted && !rep.failure()) {
            add("image", "Image", StepStatus::Failed, "The worker image " + image + " cannot run the worker on " + on + ": it does not start.", ask);
        }
        // the node's own words, the last of them
        const std::size_t from = raw.size() > 40 ? raw.size() - 40 : 0;
        for (std::size_t i = from; i < raw.size(); ++i) rep.output += raw[i] + "\n";
        rep.output = trim(rep.output);
        return rep;
    }

    // --- the session's public face ---------------------------------------------------

    Session::Session() : impl_(std::make_unique<Impl>()) {}

    Session::~Session() {
        impl_->stopKeeperThread();
        impl_->stopWorkerThread();
        impl_->stopBuilderThread();
        {
            const std::lock_guard<std::mutex> g(impl_->controlMutex);
            impl_->control.reset();
        }
        auto s = impl_->sshSession();
        if (s) s->close();
    }

    void Session::setPrompt(PromptFn fn) { impl_->prompt = std::move(fn); }
    void Session::setAskpassProgram(std::string program) { impl_->askpassProgram = std::move(program); }
    void Session::setChanged(std::function<void()> fn) { impl_->changed = std::move(fn); }
    void Session::setLog(std::function<void(const std::string&)> fn) { impl_->log = std::move(fn); }
    void Session::setPollInterval(std::chrono::milliseconds queue, std::chrono::milliseconds keepAlive) {
        impl_->queuePoll = queue;
        impl_->keepAlive = keepAlive;
    }

    void Session::connectJob(const Profile& profile) { start(profile, Mode::Job); }
    void Session::startWorker(const Profile& profile) { start(profile, Mode::Worker); }
    void Session::connect(const Profile& profile) { start(profile, Mode::Both); }
    void Session::logIn(const Profile& profile) { start(profile, Mode::Login); }

    void Session::adoptJob(const Profile& profile, const std::string& jobId) {
        if (trim(jobId).empty()) return;
        {
            const std::lock_guard<std::mutex> g(impl_->m);
            impl_->adoptId = trim(jobId);
        }
        start(profile, Mode::Adopt);
    }

    std::vector<ClusterJob> Session::listJobs() {
        auto s = impl_->sshSession();
        if (!s || !s->isOpen()) throw ssh::SshError("not logged in to the cluster: connect first");
        const ssh::CommandResult r = s->run(userJobsScript(), std::chrono::seconds(60));
        if (!r.ok() && trim(r.out).empty()) throw ssh::SshError("squeue did not list your jobs", trim(r.err));
        return parseUserJobs(r.out);
    }

    void Session::start(const Profile& profile, Mode mode) {
        {
            // a held job takes the worker; a new job only once it is let go (disconnect)
            const State now = status().state;
            const bool held = now == State::JobReady || now == State::Connected;
            if (mode == Mode::Worker ? !held : held) return;
        }
        if (impl_->connecting.exchange(true)) return;
        // The previous attempt's thread first: it clears `connecting` before it
        // publishes its state, and may still start the keeper after that.
        if (impl_->worker.joinable()) impl_->worker.join();
        impl_->stopKeeperThread();
        if (mode != Mode::Worker) {
            const std::lock_guard<std::mutex> g(impl_->controlMutex);
            impl_->control.reset();
        }
        impl_->cancel.store(false);
        std::string previousJob;
        impl_->update([&](Status& x) {
            if (mode == Mode::Worker) {
                x.state = State::Starting;
                x.since = std::chrono::steady_clock::now();
                x.reason.clear();
                x.dropped = false;
                return;
            }
            const bool sshUp = x.sshUp;
            const std::string host = x.host;
            const std::string home = x.home;
            // a job left running (disconnect without cancelling it, or a drop) on this host
            const bool left = !x.jobId.empty() && x.host == profile.host && (x.state == State::Disconnected || x.state == State::Idle);
            if (mode != Mode::Login && mode != Mode::Adopt && left) previousJob = x.jobId;
            const std::string keptJob = mode == Mode::Login && left ? x.jobId : std::string();
            const BuildStatus build = x.build;
            x = Status{};
            x.jobId = keptJob;   // a login keeps it for the next Connect
            x.state = State::Connecting;
            x.since = std::chrono::steady_clock::now();
            x.sshUp = sshUp;
            x.host = host;
            x.home = home;
            x.build = build.phase == BuildStatus::Phase::Probing || build.phase == BuildStatus::Phase::Building ? build : BuildStatus{};
        });
        {
            const std::lock_guard<std::mutex> g(impl_->m);
            impl_->profile = profile;
            impl_->mode = mode;
            impl_->reattachJob = previousJob;
            if (mode != Mode::Adopt) impl_->adoptId.clear();
            if (mode != Mode::Worker && previousJob.empty()) {
                // a new job: nothing of the last one's worker to reattach to
                impl_->workerLog.clear();
                impl_->workerPort = 0;
            }
        }
        impl_->worker = std::thread([this] { impl_->run(); });
    }

    void Session::stopWorker() {
        if (impl_->connecting.load()) return;
        std::string id;
        {
            const std::lock_guard<std::mutex> g(impl_->m);
            if (impl_->status.state != State::Connected) return;
            id = impl_->status.jobId;
        }
        impl_->stopKeeperThread();
        impl_->stopWorkerStep(id);
        impl_->update([&](Status& x) {
            x.state = State::JobReady;
            x.since = std::chrono::steady_clock::now();
            x.reason = "the worker was stopped";
            for (const Step s : {Step::Checks, Step::Start, Step::Hello}) x.steps[static_cast<std::size_t>(s)] = StepState{};
        });
        impl_->say("HPC: the worker was stopped; job " + id + " is still held");
        impl_->startKeeper();
    }

    void Session::cancelJob() {
        impl_->stopKeeperThread();
        impl_->stopWorkerThread();
        impl_->stopBuilderThread();
        {
            const std::lock_guard<std::mutex> g(impl_->controlMutex);
            impl_->control.reset();
        }
        std::string id, note;
        bool ended = false, adopted = false;
        {
            const std::lock_guard<std::mutex> g(impl_->m);
            id = impl_->status.jobId;
            adopted = impl_->status.adopted;
        }
        auto s = impl_->sshSession();
        const bool up = s && s->isOpen();
        if (!id.empty() && adopted) {
            // the user's own job: its worker step ends, the job runs on
            if (up) impl_->stopWorkerStep(id);
            note = "job " + id + " was yours before SIRIUS: it keeps running";
        } else if (!id.empty() && up) {
            try {
                const ssh::CommandResult r = s->run("scancel " + id, std::chrono::seconds(30));
                ended = r.ok();
                note = ended ? "job " + id + " cancelled" : "scancel " + id + " failed: " + trim(r.err);
            } catch (const std::exception& e) {
                note = "scancel " + id + " failed: " + e.what();
            }
        }
        if (ended || adopted) {
            const std::lock_guard<std::mutex> g(impl_->m);
            impl_->workerLog.clear();
            impl_->workerPort = 0;
            impl_->reattachJob.clear();
            impl_->adoptedJob.clear();
        }
        impl_->update([&](Status& x) {
            x.state = up ? State::Idle : State::Disconnected;
            x.since = std::chrono::steady_clock::now();
            x.reason = note;
            x.remoteOutput.clear();
            x.fix.clear();
            x.dropped = false;
            x.sshUp = up;
            x.caps = WorkerCapabilities{};
            x.engineBuild.clear();
            x.engineBuildNote.clear();
            x.nodeChecks.clear();
            for (std::size_t i = 1; i < x.steps.size(); ++i) x.steps[i] = StepState{};
            if (ended || adopted) {
                x.jobId.clear();
                x.node.clear();
                x.jobState.clear();
                x.jobLimitSeconds = -2;
                x.jobStarted = {};
                x.jobEnded = ended;
                x.adopted = false;
            }
        });
        impl_->say("HPC: " + (note.empty() ? std::string("no job to cancel") : note) + (up ? "; still logged in" : std::string()));
    }

    void Session::buildImage(const Profile& profile, const std::string& defFile, const std::string& image) {
        const Status st = status();
        const bool held = st.state == State::JobReady || st.state == State::Connected;
        if (!held || st.build.phase == BuildStatus::Phase::Probing || st.build.phase == BuildStatus::Phase::Building) return;
        if (impl_->builder.joinable()) impl_->builder.join();
        impl_->buildCancel.store(false);
        impl_->builder = std::thread([this, profile, defFile, image, id = st.jobId] { impl_->build(profile, id, defFile, image, false); });
    }

    void Session::probeBuild(const Profile& profile) {
        const Status st = status();
        const bool held = st.state == State::JobReady || st.state == State::Connected;
        if (!held || st.build.phase == BuildStatus::Phase::Probing || st.build.phase == BuildStatus::Phase::Building) return;
        if (impl_->builder.joinable()) impl_->builder.join();
        impl_->buildCancel.store(false);
        impl_->builder = std::thread([this, profile, id = st.jobId] { impl_->build(profile, id, {}, {}, true); });
    }

    void Session::cancelBuild() { impl_->buildCancel.store(true); }

    void Session::cancelConnect() {
        impl_->cancel.store(true);
        impl_->abortLogin.store(true);
    }

    void Session::disconnect(bool cancelJob) {
        impl_->stopKeeperThread();
        impl_->stopWorkerThread();
        impl_->stopBuilderThread();
        {
            const std::lock_guard<std::mutex> g(impl_->controlMutex);
            impl_->control.reset();
        }
        std::string id, note;
        bool ended = false, adopted = false;
        {
            const std::lock_guard<std::mutex> g(impl_->m);
            id = impl_->status.jobId;
            adopted = impl_->status.adopted;
        }
        auto s = impl_->sshSession();
        if (!id.empty()) {
            if (adopted) {
                note = "job " + id + " was yours before SIRIUS: it keeps running";
            } else if (cancelJob && s && s->isOpen()) {
                try {
                    const ssh::CommandResult r = s->run("scancel " + id, std::chrono::seconds(30));
                    note = r.ok() ? "job " + id + " cancelled" : "scancel " + id + " failed: " + trim(r.err);
                    if (r.ok()) {
                        const std::lock_guard<std::mutex> g(impl_->m);
                        impl_->status.jobId.clear();
                        impl_->workerLog.clear();
                        ended = true;
                    }
                } catch (const std::exception& e) {
                    note = "scancel " + id + " failed: " + e.what();
                }
            } else {
                note = "job " + id + " left running";
            }
        }
        if (s) s->close();
        impl_->update([&](Status& x) {
            x.state = State::Disconnected;
            x.since = std::chrono::steady_clock::now();
            x.reason = "disconnected" + (note.empty() ? std::string() : " (" + note + ")");
            x.jobEnded = x.jobEnded || ended;
            x.dropped = false;
            x.sshUp = false;
        });
        {
            const std::lock_guard<std::mutex> g(impl_->m);
            impl_->ssh.reset();
        }
        impl_->say("HPC: disconnected" + (note.empty() ? std::string() : ": " + note));
    }

    Status Session::status() const {
        const std::lock_guard<std::mutex> g(impl_->m);
        Status s = impl_->status;
        if (impl_->ssh) s.sshUp = impl_->ssh->isOpen();
        return s;
    }

    Profile Session::profile() const {
        const std::lock_guard<std::mutex> g(impl_->m);
        return impl_->profile;
    }

    bool Session::connected() const { return status().state == State::Connected; }
    bool Session::sshUp() const { return status().sshUp; }

    bool Session::hasJobRunning() const {
        const State s = status().state;
        return s == State::JobReady || s == State::Starting || s == State::Connected;
    }

    bool Session::hasJob() const {
        const std::lock_guard<std::mutex> g(impl_->m);
        return !impl_->status.jobId.empty();
    }

    Session::Endpoint Session::endpoint() const {
        const std::lock_guard<std::mutex> g(impl_->m);
        Endpoint e;
        if (impl_->status.state != State::Connected) return e;
        e.host = impl_->status.node;
        e.port = impl_->workerPort;
        e.token = impl_->token;
        e.socksPort = impl_->ssh ? impl_->ssh->socksPort() : 0;
        return e;
    }

    std::unique_ptr<RemoteWorker> Session::connectWorker(std::chrono::milliseconds timeout, const std::function<bool()>& cancelled) const {
        const Endpoint e = endpoint();
        if (e.host.empty()) throw ProtocolError("not connected to a cluster worker (Process \xE2\x96\xB8 Connect to cluster\xE2\x80\xA6)");
        return RemoteWorker::connect(e.host, e.port, e.token, timeout, cancelled, e.socksPort);
    }

    Listing Session::list(const std::string& path, int maxEntries) {
        auto s = impl_->sshSession();
        if (!s || !s->isOpen()) throw ssh::SshError("not logged in to the cluster: connect first");
        const ssh::CommandResult r = s->run(listingScript(path, maxEntries), std::chrono::seconds(60));
        std::string line;
        std::istringstream in(r.out);
        std::string l;
        while (std::getline(in, l))
            if (!l.empty() && l.front() == '{') line = l;
        if (line.empty()) throw ssh::SshError("the cluster did not list " + path, trim(r.err));
        return parseListing(line);
    }

    std::optional<ClusterInfo> Session::clusterInfo() const {
        const std::lock_guard<std::mutex> g(impl_->m);
        return impl_->info;
    }

    ClusterInfo Session::refreshClusterInfo() {
        std::string host;
        {
            const std::lock_guard<std::mutex> g(impl_->m);
            host = impl_->status.host;
        }
        return impl_->queryInfo(host, {});
    }

    bool Session::queryingClusterInfo() const { return impl_->querying.load(); }

    ssh::CommandResult Session::run(const std::string& script, std::chrono::milliseconds timeout) {
        auto s = impl_->sshSession();
        if (!s || !s->isOpen()) throw ssh::SshError("not logged in to the cluster: connect first");
        return s->run(script, timeout);
    }

} // namespace sirius::app::cluster
