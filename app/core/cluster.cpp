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
            case Step::Checks: return "Check the image";
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
        std::string reattachJob;                     // a job left running at the last disconnect, to reattach to, under m
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
            }
            update([&](Status& x) { x.jobId = id; });
            const std::string where = p.partition.empty() ? std::string() : " to " + p.partition;
            stepState(Step::Submit, StepStatus::Done, "job " + id + where);
            say("Cluster: submitted job " + id + where);
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

        // Checks for the worker: srun, the checkout (the worker's code is
        // bound into the image from it), the image, the launcher, the data
        // folders, sirius and numpy importable in the image, the engine
        // build that fits this application, and the node cache folder.
        void checks(const Profile& p) {
            if (trim(p.container).empty())
                throw Failure{Step::Checks, "No worker image is set: SIRIUS's worker runs in an Apptainer/Singularity image (.sif) on the cluster.", {}, "Pick the image under Worker image (Browse lists the cluster's files), or build one with \"Build an image\". "
                                                                                                                                                         "app/python/slurm/README.md says how images are made."};
            if (trim(p.checkout).empty())
                throw Failure{Step::Checks, "No SIRIUS checkout on the cluster is set, and " + p.host + " did not say where your home folder is.", {}, "Set \"SIRIUS checkout on the cluster\" under Job \xE2\x96\xB8 More options to the folder you cloned SIRIUS into there."};
            stepState(Step::Checks, StepStatus::Running, "the checkout, the image, the launcher");
            const std::string co = remotePathWord(p.checkout);
            std::string script;
            script += "command -v srun >/dev/null 2>&1 && echo srun=yes || echo srun=no\n";
            script += "[ -f " + co + "/app/python/sirius_worker/__main__.py ] && echo worker=yes || echo worker=no\n";
            script += "[ -f " + co + "/app/python/slurm/sirius_worker.sbatch ] && echo template=yes || echo template=no\n";
            script += "echo \"home=$HOME\"\n";
            script += "C=" + remotePathWord(p.container) + "\n";
            script += "if [ ! -f \"$C\" ]; then echo image=no; elif test -r \"$C\"; then echo image=yes; else echo image=unreadable; fi\n";
            // each bind's host path: apptainer stops before the worker starts on one that is not there
            const std::vector<std::string> binds = bindHostPaths(p.bind);
            for (std::size_t i = 0; i < binds.size(); ++i)
                script += "test -d " + remotePathWord(binds[i]) + " && echo bind" + std::to_string(i) + "=yes || echo bind" + std::to_string(i) + "=no\n";
            const bool builds = p.engine && trim(p.engineBin).empty() && !trim(p.engineBuilds).empty();
            if (builds) script += engineBuildsScript(p.engineBuilds);
            script += launcherScript(p);
            script += "echo \"launcher=$L\"\n";
            const std::string bw = bindWord(p.bind);
            script += "if test -r \"$C\" && [ -n \"$L\" ]; then\n"
                      // a folder named sirius (a checkout in the working directory) imports as an empty namespace: not the package
                      "    \"$L\" exec " +
                      (bw.empty() ? std::string() : "--bind " + bw + " ") +
                      "\"$C\" python -c 'import sys, importlib.util as u; import sirius, numpy; "
                      "assert getattr(sirius, \"__file__\", None), \"sirius is only a folder named so, not the compiled package\"; print(\"c_pyver=%d.%d\" % "
                      "sys.version_info[:2]); print(\"c_torch=%s\" % (\"yes\" if u.find_spec(\"torch\") else \"no\"))' </dev/null\n"
                      "    echo \"c_rc=$?\"\n"
                      "fi\n";
            // a first exec of a large image on a network file system takes a while
            const ssh::CommandResult r = remote(script, std::chrono::seconds(240));
            auto kv = keyValues(r.out);
            const std::string ask = "Build the worker image again (\"Build an image\", or app/python/slurm/README.md), or ask whoever builds it on this "
                                    "cluster for the current one, and pick it under Worker image.";
            if (kv["srun"] != "yes")
                throw Failure{Step::Checks, "There is no srun on " + p.host + ": the worker starts in the job with srun.", trim(r.err),
                              "Connect to the cluster's login node, the one you submit jobs from."};
            if (kv["worker"] != "yes" || kv["template"] != "yes")
                throw Failure{Step::Checks, "There is no SIRIUS checkout at " + p.checkout + " on " + p.host + " (it needs app/python/sirius_worker and app/python/slurm).",
                              trim(r.err),
                              "Clone this SIRIUS repository there (the same version as this application), or set \"SIRIUS checkout on the cluster\" under "
                              "Job \xE2\x96\xB8 More options to where it is."};
            if (!trim(kv["home"]).empty()) update([&](Status& x) { x.home = trim(kv["home"]); });
            if (kv["image"] == "unreadable")
                throw Failure{Step::Checks, "The worker image " + p.container + " on " + p.host + " cannot be read (test -r failed).", trim(r.err),
                              "Make it readable to you (chmod a+r), or pick a copy you can read under Worker image."};
            if (kv["image"] != "yes") throw Failure{Step::Checks, "There is no worker image at " + p.container + " on " + p.host + ".", trim(r.err), ask};
            for (std::size_t i = 0; i < binds.size(); ++i)
                if (kv["bind" + std::to_string(i)] != "yes")
                    throw Failure{Step::Checks, "The data folder " + binds[i] + " is not a folder on " + p.host + ": apptainer would stop before the worker starts.",
                                  trim(r.err), "Correct it or remove it under Data folders (the folders on the cluster your datasets live in)."};
            if (kv["launcher"].empty())
                throw Failure{Step::Checks,
                              "Neither " + (p.launcher.empty() ? std::string("apptainer") : p.launcher) + " nor singularity can be run on " + p.host +
                                  " (module load was tried).",
                              trim(r.err),
                              "Ask your cluster's support how to run apptainer there, or set \"Container launcher\" under Job \xE2\x96\xB8 More options to its full path."};
            if (kv["c_rc"] != "0")
                throw Failure{Step::Checks, "The worker image " + p.container + " cannot run the worker: sirius and numpy do not import in it.", trim(r.err), ask};
            std::string detail = "checkout \xC2\xB7 " + kv["launcher"] + " \xC2\xB7 image: python " + kv["c_pyver"] + ", sirius, numpy";
            detail += kv["c_torch"] == "yes" ? ", torch" : " \xC2\xB7 no torch (models will not run)";
            if (!binds.empty()) detail += " \xC2\xB7 " + std::to_string(binds.size()) + (binds.size() == 1 ? " data folder" : " data folders");
            // the engine build that serves this application
            std::string buildDir, buildNote;
            if (builds) {
                if (kv["builds"] != "yes")
                    throw Failure{Step::Checks, "There is no engine builds folder at " + p.engineBuilds + " on " + p.host + ".", trim(r.err), engineBuildFix(p)};
                const std::vector<EngineBuild> found = parseEngineBuilds(r.out);
                const int pick = pickEngineBuild(found, buildInfo(), &buildNote);
                if (pick < 0) throw Failure{Step::Checks, buildNote, listBuilds(found), engineBuildFix(p)};
                buildDir = trim(p.engineBuilds);
                while (buildDir.size() > 1 && buildDir.back() == '/') buildDir.pop_back();
                buildDir += "/" + found[static_cast<std::size_t>(pick)].dir;
                detail += " \xC2\xB7 engine " + found[static_cast<std::size_t>(pick)].dir.substr(0, 12) + " (" + buildNote + ")";
            } else if (p.engine) {
                buildNote = trim(p.engineBin).empty() ? std::string("the image's own engine") : "engine " + trim(p.engineBin);
                detail += " \xC2\xB7 " + buildNote;
            }
            {
                const std::lock_guard<std::mutex> g(m);
                engineDir = buildDir;
            }
            update([&](Status& x) {
                x.engineBuild = buildDir;
                x.engineBuildNote = buildNote;
            });
            if (const std::string w = emptyBindWarning(p); !w.empty()) {
                // not a failure: data in the home folder opens all the same
                stepState(Step::Checks, StepStatus::Warning, detail + " \xC2\xB7 " + w);
                say("Cluster: " + w);
            } else {
                stepState(Step::Checks, StepStatus::Done, detail);
            }
            checkScratch(p);
        }

        static std::string engineBuildFix(const Profile& p) {
            return "Build the engine of this SIRIUS (commit " + buildInfo().commit.substr(0, 12) + ") into " +
                   (trim(p.engineBuilds).empty() ? std::string("the engine builds folder") : p.engineBuilds) +
                   " -- app/python/slurm/README.md, \"Engine builds\", has the command -- or clear Engine builds under Job \xE2\x96\xB8 More options "
                   "to use the image's own engine.";
        }

        static std::string listBuilds(const std::vector<EngineBuild>& found) {
            std::string out;
            for (const EngineBuild& b : found)
                out += b.dir + ": " + (b.readable ? b.info.build + (b.runnable ? std::string() : " (bin/sirius-cli missing)") : std::string("BUILD.json unreadable")) +
                       "\n";
            return out;
        }

        // The node cache folder (the engine's --scratch), when one is set: a
        // folder there you can write to, or one that can be made. One that
        // is not on the login node at all may be a node's own disk: said,
        // not refused.
        void checkScratch(const Profile& p) {
            const std::string dir = trim(p.scratch);
            if (!p.engine || dir.empty()) return;
            std::string script = "S=" + remotePathWord(dir) + "\n";
            script += "if [ -d \"$S\" ]; then if [ -w \"$S\" ]; then echo scratch=yes; else echo scratch=readonly; fi\n"
                      "elif [ -e \"$S\" ]; then echo scratch=file\n"
                      "else d=\"$S\"; while [ ! -e \"$d\" ] && [ \"$d\" != / ] && [ \"$d\" != . ]; do d=$(dirname \"$d\"); done\n"
                      "     if [ -d \"$d\" ] && [ -w \"$d\" ]; then echo scratch=creatable; else echo scratch=missing; fi\n"
                      "fi\n";
            const ssh::CommandResult r = remote(script, std::chrono::seconds(30));
            const std::string s = keyValues(r.out)["scratch"];
            const std::string where = "Choose another folder under Job \xE2\x96\xB8 More options, or leave it empty to use the node's temporary folder.";
            if (s == "file") throw Failure{Step::Checks, "The node cache folder " + dir + " is a file on " + p.host + ", not a folder.", trim(r.err), where};
            if (s == "readonly") throw Failure{Step::Checks, "You cannot write to the node cache folder " + dir + " on " + p.host + ".", trim(r.err), where};
            StepState now;
            {
                const std::lock_guard<std::mutex> g(m);
                now = status.steps[static_cast<std::size_t>(Step::Checks)];
            }
            if (s == "yes" || s == "creatable") {
                stepState(Step::Checks, now.status, now.detail + " \xC2\xB7 node cache " + dir);
                return;
            }
            const std::string w = "the node cache folder " + dir + " is not on the login node: fine if it is a disk of the compute nodes only";
            stepState(Step::Checks, StepStatus::Warning, now.detail + " \xC2\xB7 " + w);
            say("Cluster: " + w);
        }

        // The worker as a step of the held job: app/python/slurm/
        // sirius_worker.sbatch (bash, not sbatch) under srun --overlap, in
        // the background and in a session of its own, so it outlives this
        // command and the SSH connection. The token goes to a 0600 file whose
        // name is all the step is given.
        void startStep(const Profile& p, const std::string& id) {
            stepState(Step::Start, StepStatus::Running, "srun --jobid=" + id);
            std::string dir;
            int gpus = 0;
            int seq = 0;
            {
                const std::lock_guard<std::mutex> g(m);
                dir = engineDir;
                gpus = jobProfile.gpus;
                seq = ++workerSeq;
                token = ssh::randomHex(16);
            }
            const std::string logPath = "\"$HOME/.sirius/run/sirius-worker-" + id + "-" + std::to_string(seq) + ".log\"";
            std::string script = "umask 077\n";
            script += "mkdir -p \"$HOME/.sirius/run\" && chmod 700 \"$HOME/.sirius/run\" || exit 4\n";
            // token files of workers that never started, a day old
            script += "find \"$HOME/.sirius/run\" -maxdepth 1 -name 'token.*' -mmin +1440 -exec rm -f {} + 2>/dev/null\n";
            script += "cd " + remotePathWord(p.checkout) + " || exit 3\n";
            // The token is written to a private file over this command
            // channel (printf is a builtin: no argument list shows it) and the
            // step is told only the file's name. The worker reads it and deletes it.
            script += "tf=$(mktemp \"$HOME/.sirius/run/token.XXXXXXXX\") || exit 4\n";
            script += "printf '%s' " + shellQuote(token) + " > \"$tf\" || { rm -f \"$tf\"; exit 4; }\n";
            script += "unset SIRIUS_TOKEN SIRIUS_VENV\nexport SIRIUS_TOKEN_FILE=\"$tf\"\n";
            // port 0: the worker takes a free one and says which in its log
            script += "export SIRIUS_PORT=0 SIRIUS_MAX_CLIENTS=8\n";
            script += "export SIRIUS_CONTAINER=" + remotePathWord(p.container) + "\n";
            script += "export SIRIUS_LAUNCHER=" + shellQuote(p.launcher.empty() ? std::string("apptainer") : p.launcher) + "\n";
            // the data folders and the extra PYTHONPATH, quoted (a "~/" bind made $HOME): never the token
            if (const std::string bw = bindWord(p.bind); !bw.empty()) script += "export SIRIUS_CONTAINER_BIND=" + bw + "\n";
            else script += "unset SIRIUS_CONTAINER_BIND\n";
            if (const std::string pp = trim(p.containerPythonPath); !pp.empty()) script += "export SIRIUS_CONTAINER_PYTHONPATH=" + shellQuote(pp) + "\n";
            else script += "unset SIRIUS_CONTAINER_PYTHONPATH\n";
            script += gpus <= 0 ? "export SIRIUS_DEVICE=cpu\n" : "unset SIRIUS_DEVICE\n";
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
            script += "S=; command -v setsid >/dev/null 2>&1 && S=setsid\n";
            std::string srun = "srun --jobid=" + id + " --overlap --nodes=1 --ntasks=1 --job-name=sirius-worker";
            if (gpus > 0) srun += " --gres=gpu:" + std::to_string(gpus);
            script += "$S nohup " + srun + " bash app/python/slurm/sirius_worker.sbatch > " + logPath + " 2>&1 < /dev/null &\n";
            script += "echo \"srun=$!\"\n";
            const ssh::CommandResult r = remote(script, std::chrono::seconds(60));
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
                throw Failure{Step::Hello, "Could not reach the worker on " + node + ":" + std::to_string(port) + " through the SSH tunnel.", e.what(), {}};
            }
            w->setCancelGrace(std::chrono::milliseconds(0));
            const WorkerCapabilities caps = w->capabilities();
            // SIRIUS's engine serves this application only when their
            // operations are the same (core/build_info.hpp): refused here, in
            // words, before anything is run on it.
            if (caps.engine.is_object())
                if (const std::string refusal = engineMismatch(buildInfo(), buildInfoFromJson(caps.engine)); !refusal.empty())
                    throw Failure{Step::Hello, refusal, {}, engineBuildFix(p)};
            int kinds = 0;
            for (const std::string& meth : caps.methods)
                if (meth.rfind("run:", 0) == 0) ++kinds;
            {
                const std::lock_guard<std::mutex> g(controlMutex);
                control = std::move(w);
            }
            update([&](Status& x) { x.caps = caps; });
            StepStatus helloStatus = StepStatus::Done;
            std::string detail;
            if (caps.engine.is_object()) {
                const json python = caps.engine.value("python", json::object());
                detail = "SIRIUS engine " + caps.engine.value("build", caps.version) + " \xC2\xB7 " + caps.device + " \xC2\xB7 Python worker " +
                         python.value("state", std::string("disabled")) + " \xC2\xB7 session " + caps.engine.value("session", std::string());
            } else if (p.engine) {
                // asked for the engine, got the Python worker (an old image): built-in steps will be refused
                helloStatus = StepStatus::Warning;
                detail = "sirius_worker " + caps.version + " \xC2\xB7 " + caps.device + " \xC2\xB7 no SIRIUS engine in this worker: only the Python steps run there";
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
            std::string again;
            {
                const std::lock_guard<std::mutex> g(m);
                again = reattachJob;
                reattachJob.clear();
            }
            if (!again.empty() && jobRunning(again)) {
                update([&](Status& x) { x.jobId = again; });
                stepState(Step::Submit, StepStatus::Done, "job " + again + " \xC2\xB7 reattached");
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
            });
            stopWorkerStep(id);   // a new image, new data folders: the old worker goes first
            checks(p);
            startStep(p, id);
            waitForWorker(p, id, node);
            hello(p, node);
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
                    if (how == Mode::Job) {
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

        void keep() {
            int tick = 0;
            for (;;) {
                {
                    std::unique_lock<std::mutex> lk(keeperMutex);
                    if (keeperWake.wait_for(lk, keepAlive, [this] { return stopKeeper.load(); })) return;
                }
                ++tick;
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
                {
                    const std::lock_guard<std::mutex> g(controlMutex);
                    if (control) {
                        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
                        try {
                            control->call("ping", json::object(), {}, {}, [&] { return stopKeeper.load() || std::chrono::steady_clock::now() > deadline; });
                        } catch (const std::exception& e) {
                            if (stopKeeper.load()) return;
                            pingFailed = true;
                            why = isCancellation(e) ? std::string("no answer within 20 s") : std::string(e.what());
                        }
                    }
                }
                if (pingFailed || tick % 2 == 0) {
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

    // --- the engine builds ----------------------------------------------------------

    std::string engineBuildsScript(const std::string& folder) {
        // the newest first; each line: "@@build <dir> <yes|no> <BUILD.json on one line>"
        return "EB=" + remotePathWord(folder) +
               "\n"
               "if [ -d \"$EB\" ]; then echo builds=yes\n"
               "  for d in $(ls -1t \"$EB\" 2>/dev/null); do\n"
               "    [ -f \"$EB/$d/BUILD.json\" ] || continue\n"
               "    x=no; [ -x \"$EB/$d/bin/sirius-cli\" ] && x=yes\n"
               "    printf '@@build %s %s ' \"$d\" \"$x\"; tr -d '\\n\\r' < \"$EB/$d/BUILD.json\"; echo\n"
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
            std::istringstream words(line.substr(8));
            EngineBuild b;
            std::string runnable;
            words >> b.dir >> runnable;
            b.runnable = runnable == "yes";
            std::string rest;
            std::getline(words, rest);
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
        // this application's own commit first
        for (std::size_t i = 0; i < builds.size(); ++i) {
            const EngineBuild& b = builds[i];
            if (b.runnable && b.readable && !app.commit.empty() && app.commit != "unknown" && b.info.commit == app.commit &&
                engineMismatch(app, b.info).empty()) {
                if (note) *note = "this build";
                return static_cast<int>(i);
            }
        }
        // else the newest with the same operations and engine API
        for (std::size_t i = 0; i < builds.size(); ++i) {
            const EngineBuild& b = builds[i];
            if (b.runnable && b.readable && engineMismatch(app, b.info).empty()) {
                if (note) *note = "the same operations as this build";
                return static_cast<int>(i);
            }
        }
        if (note) {
            if (builds.empty()) *note = "The engine builds folder holds no build (<commit>/BUILD.json and bin/sirius-cli).";
            else
                *note = "None of the " + std::to_string(builds.size()) + (builds.size() == 1 ? " engine build" : " engine builds") +
                        " fits this application (" + app.build + "): their operations or engine API differ.";
        }
        return -1;
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
            if (mode != Mode::Login && left) previousJob = x.jobId;
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
        bool ended = false;
        {
            const std::lock_guard<std::mutex> g(impl_->m);
            id = impl_->status.jobId;
        }
        auto s = impl_->sshSession();
        const bool up = s && s->isOpen();
        if (!id.empty() && up) {
            try {
                const ssh::CommandResult r = s->run("scancel " + id, std::chrono::seconds(30));
                ended = r.ok();
                note = ended ? "job " + id + " cancelled" : "scancel " + id + " failed: " + trim(r.err);
            } catch (const std::exception& e) {
                note = "scancel " + id + " failed: " + e.what();
            }
        }
        if (ended) {
            const std::lock_guard<std::mutex> g(impl_->m);
            impl_->workerLog.clear();
            impl_->workerPort = 0;
            impl_->reattachJob.clear();
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
            for (std::size_t i = 1; i < x.steps.size(); ++i) x.steps[i] = StepState{};
            if (ended) {
                x.jobId.clear();
                x.node.clear();
                x.jobState.clear();
                x.jobLimitSeconds = -2;
                x.jobStarted = {};
                x.jobEnded = true;
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
        bool ended = false;
        {
            const std::lock_guard<std::mutex> g(impl_->m);
            id = impl_->status.jobId;
        }
        auto s = impl_->sshSession();
        if (!id.empty()) {
            if (cancelJob && s && s->isOpen()) {
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
