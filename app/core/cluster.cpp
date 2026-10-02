#include "core/cluster.hpp"

#include "core/cancel.hpp"
#include "core/errors.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
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

    json Profile::toJson() const {
        json j = {{"host", host}, {"checkout", checkout}, {"venv", venv}, {"container", container}, {"launcher", launcher}, {"bind", bind}, {"containerPythonPath", containerPythonPath}, {"partition", partition}, {"account", account}, {"qos", qos}, {"time", time}, {"gpus", gpus}, {"cpus", cpus}, {"mem", mem}, {"port", port}, {"ssh", sshProgram}};
        json hosts = json::object();
        for (const auto& [h, c] : perHost) {
            json one = {{"partition", c.partition}, {"account", c.account}, {"qos", c.qos}, {"time", c.time}};
            if (c.bind) one["bind"] = *c.bind;
            if (c.containerPythonPath) one["containerPythonPath"] = *c.containerPythonPath;
            hosts[h] = one;
        }
        j["perHost"] = hosts;
        return j;
    }

    Profile Profile::fromJson(const json& j) {
        Profile p;
        if (!j.is_object()) return p;
        auto str = [&](const char* k, std::string& out) {
            if (j.contains(k) && j[k].is_string()) out = j[k].get<std::string>();
        };
        auto num = [&](const char* k, int& out) {
            if (j.contains(k) && j[k].is_number_integer()) out = j[k].get<int>();
        };
        str("host", p.host);
        str("checkout", p.checkout);
        str("venv", p.venv);
        str("container", p.container);
        str("launcher", p.launcher);
        if (p.launcher.empty()) p.launcher = "apptainer";
        str("bind", p.bind);
        str("containerPythonPath", p.containerPythonPath);
        str("partition", p.partition);
        str("account", p.account);
        str("qos", p.qos);
        str("time", p.time);
        num("gpus", p.gpus);
        num("cpus", p.cpus);
        str("mem", p.mem);
        num("port", p.port);
        str("ssh", p.sshProgram);
        if (j.contains("perHost") && j["perHost"].is_object())
            for (const auto& [h, c] : j["perHost"].items()) {
                if (!c.is_object()) continue;
                SlurmChoice s;
                auto field = [&](const char* k, std::string& out) {
                    if (c.contains(k) && c[k].is_string()) out = c[k].get<std::string>();
                };
                field("partition", s.partition);
                field("account", s.account);
                field("qos", s.qos);
                field("time", s.time);
                if (c.contains("bind") && c["bind"].is_string()) s.bind = c["bind"].get<std::string>();
                if (c.contains("containerPythonPath") && c["containerPythonPath"].is_string())
                    s.containerPythonPath = c["containerPythonPath"].get<std::string>();
                p.perHost[h] = s;
            }
        return p;
    }

    const char* stepTitle(Step s) {
        switch (s) {
            case Step::Login: return "SSH login";
            case Step::Checks: return "Checks on the cluster";
            case Step::Submit: return "Submit the worker job";
            case Step::Queue: return "Wait in the queue";
            case Step::Start: return "Worker starting on the node";
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
        return "the container sees only the image and your home folder: datasets elsewhere (e.g. /clusterfs) will not open "
               "\xE2\x80\x94 add them under Bind";
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
        for (const std::string& b : bindHostPaths(p.bind)) roots.push_back(expandedPath(b, h));
        for (const std::string& r : roots)
            if (isWithin(path, r)) return {};
        const std::size_t second = path.find('/', 1);
        const std::string top = second == std::string::npos ? path : path.substr(0, second);
        return "This path is not bound into the worker's container: add it under Bind (e.g. " + top + ") and reconnect.";
    }

    std::string clusterInfoScript() {
        // Fixed text only: nothing of the profile is in it. $u is the
        // cluster's own name for this user; `timeout` keeps a slurmdbd that
        // does not answer from holding the command channel.
        return "u=\"${USER:-$(id -un 2>/dev/null)}\"\n"
               "T=; command -v timeout >/dev/null 2>&1 && T='timeout 30'\n"
               "echo \"@@user $u\"\n"
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
        std::string token;
        int workerPort = 0;                          // what the worker announced, under m
        bool submitJob = true;                       // false: log in and list the partitions only (logIn), under m
        std::optional<ClusterInfo> info;             // under m
        std::atomic<bool> querying{false};
        std::mutex controlMutex;
        std::unique_ptr<RemoteWorker> control;       // under controlMutex

        std::thread worker, keeper;
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
                              "Check the host, user and password, then press Connect again (a failed login is never retried)."};
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

        // Checks for a worker in a container image: sbatch, the checkout
        // (the worker's code is bound into the container from it), the
        // image, the launcher, and sirius and numpy importable in the
        // image. The venv plays no part.
        void checksContainer(const Profile& p) {
            stepState(Step::Checks, StepStatus::Running, "sbatch, the checkout, the container image");
            const std::string co = remotePathWord(p.checkout);
            std::string script;
            script += "command -v sbatch >/dev/null 2>&1 && echo sbatch=yes || echo sbatch=no\n";
            script += "[ -f " + co + "/app/python/sirius_worker/__main__.py ] && echo worker=yes || echo worker=no\n";
            script += "[ -f " + co + "/app/python/slurm/sirius_worker.sbatch ] && echo template=yes || echo template=no\n";
            script += "echo \"home=$HOME\"\n";
            script += "C=" + remotePathWord(p.container) + "\n";
            script += "if [ ! -f \"$C\" ]; then echo image=no; elif test -r \"$C\"; then echo image=yes; else echo image=unreadable; fi\n";
            // each bind's host path: apptainer stops before the worker starts on one that is not there
            const std::vector<std::string> binds = bindHostPaths(p.bind);
            for (std::size_t i = 0; i < binds.size(); ++i)
                script += "test -d " + remotePathWord(binds[i]) + " && echo bind" + std::to_string(i) + "=yes || echo bind" + std::to_string(i) + "=no\n";
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
            const std::string ask = "Build the SIRIUS worker image (it holds the compiled sirius package, numpy and torch), or ask whoever builds "
                                    "it for the current one, and set its path in the profile.";
            if (kv["sbatch"] != "yes")
                throw Failure{Step::Checks, "There is no sbatch on " + p.host + ": connect to the cluster's login node, the one you submit jobs from.", r.err, {}};
            if (kv["worker"] != "yes" || kv["template"] != "yes")
                throw Failure{Step::Checks, "There is no SIRIUS checkout at " + p.checkout + " on " + p.host + " (it needs app/python/sirius_worker and app/python/slurm).",
                              r.err, "Clone or copy this SIRIUS repository there (same version as this application), or set the checkout path."};
            if (!trim(kv["home"]).empty()) update([&](Status& x) { x.home = trim(kv["home"]); });
            if (kv["image"] == "unreadable")
                throw Failure{Step::Checks, "The container image " + p.container + " on " + p.host + " cannot be read (test -r failed).", r.err,
                              "Make it readable to you (chmod a+r), or set the path of a copy you can read."};
            if (kv["image"] != "yes") throw Failure{Step::Checks, "There is no container image at " + p.container + " on " + p.host + ".", r.err, ask};
            for (std::size_t i = 0; i < binds.size(); ++i)
                if (kv["bind" + std::to_string(i)] != "yes")
                    throw Failure{Step::Checks, "The bind path " + binds[i] + " is not a folder on " + p.host + ": apptainer would stop before the worker starts.", r.err,
                                  "Correct it or remove it under Bind (host paths the worker may read, comma separated)."};
            if (kv["launcher"].empty())
                throw Failure{Step::Checks, "Neither " + (p.launcher.empty() ? std::string("apptainer") : p.launcher) + " nor singularity can be run on " + p.host + " (module load was tried).",
                              r.err, "module load apptainer, or set the launcher to the full path of apptainer or singularity in the profile."};
            if (kv["c_rc"] != "0")
                throw Failure{Step::Checks, "The container image " + p.container + " cannot run the worker: sirius and numpy do not import in it.", trim(r.err),
                              ask};
            std::string detail = "sbatch \xC2\xB7 checkout \xC2\xB7 " + kv["launcher"] + " \xC2\xB7 image: python " + kv["c_pyver"] + ", sirius, numpy";
            detail += kv["c_torch"] == "yes" ? ", torch" : " \xC2\xB7 no torch (models will not run)";
            if (!binds.empty()) detail += " \xC2\xB7 binds " + std::to_string(binds.size());
            if (const std::string w = emptyBindWarning(p); !w.empty()) {
                // not a failure: data in the home folder opens all the same
                stepState(Step::Checks, StepStatus::Warning, detail + " \xC2\xB7 " + w);
                say("Cluster: " + w);
                return;
            }
            stepState(Step::Checks, StepStatus::Done, detail);
        }

        void checks(const Profile& p) {
            if (!p.container.empty()) {
                checksContainer(p);
                return;
            }
            stepState(Step::Checks, StepStatus::Running, "sbatch, the checkout, the Python environment");
            const std::string co = remotePathWord(p.checkout);
            std::string script;
            script += "command -v sbatch >/dev/null 2>&1 && echo sbatch=yes || echo sbatch=no\n";
            script += "[ -f " + co + "/app/python/sirius_worker/__main__.py ] && echo worker=yes || echo worker=no\n";
            script += "[ -f " + co + "/app/python/slurm/sirius_worker.sbatch ] && echo template=yes || echo template=no\n";
            script += "type module >/dev/null 2>&1 && module load python >/dev/null 2>&1\n";
            if (!p.venv.empty())
                script += "if [ -f " + remotePathWord(p.venv) + "/bin/activate ]; then echo venv=yes; . " + remotePathWord(p.venv) +
                          "/bin/activate; else echo venv=no; fi\n";
            script += "PY=$(command -v python3 || command -v python); echo \"python=$PY\"\n";
            script += "[ -n \"$PY\" ] && \"$PY\" -c 'import importlib.util as u, sys; print(\"pyver=%d.%d\" % sys.version_info[:2]); "
                      "[print(\"has_%s=%s\" % (m, \"yes\" if u.find_spec(m) else \"no\")) for m in (\"numpy\", \"torch\")]'\n";
            // TIFF datasets are read with the sirius package: imported, not just found (a
            // folder named sirius in the home directory would be found as a namespace package)
            script += "[ -n \"$PY\" ] && \"$PY\" -c 'import sirius; sirius.inspect_tiff' >/dev/null 2>&1 && echo has_sirius=yes || echo has_sirius=no\n";
            const ssh::CommandResult r = remote(script, std::chrono::seconds(90));
            auto kv = keyValues(r.out);
            const std::string pip = (p.venv.empty() ? std::string("python3 -m pip") : p.venv + "/bin/pip");
            const std::string req = p.checkout + "/app/python/requirements.txt";
            if (kv["sbatch"] != "yes")
                throw Failure{Step::Checks, "There is no sbatch on " + p.host + ": connect to the cluster's login node, the one you submit jobs from.", r.err, {}};
            if (kv["worker"] != "yes" || kv["template"] != "yes")
                throw Failure{Step::Checks, "There is no SIRIUS checkout at " + p.checkout + " on " + p.host + " (it needs app/python/sirius_worker and app/python/slurm).",
                              r.err, "Clone or copy this SIRIUS repository there (same version as this application), or set the checkout path."};
            if (!p.venv.empty() && kv["venv"] != "yes")
                throw Failure{Step::Checks, "There is no Python environment at " + p.venv + " on " + p.host + ".", r.err,
                              "python3 -m venv " + p.venv + " && " + p.venv + "/bin/pip install -r " + req + " && " + p.venv + "/bin/pip install " + p.checkout};
            if (kv["python"].empty()) throw Failure{Step::Checks, "There is no python3 on " + p.host + ".", r.err, "module load python, or set a venv"};
            if (kv["has_numpy"] != "yes")
                throw Failure{Step::Checks, "numpy is missing from " + (p.venv.empty() ? kv["python"] : p.venv) + ": the worker cannot start without it.", r.err,
                              pip + " install -r " + req};
            std::string detail = "sbatch \xC2\xB7 checkout \xC2\xB7 python " + kv["pyver"] + " \xC2\xB7 numpy";
            detail += kv["has_torch"] == "yes" ? " \xC2\xB7 torch" : " \xC2\xB7 no torch (models will not run)";
            if (kv["has_sirius"] != "yes") {
                // build the sirius package into the venv: its TIFF reader is the only one the worker has
                update([&](Status& x) { x.fix = pip + " install " + p.checkout; });
                stepState(Step::Checks, StepStatus::Warning, detail + " \xC2\xB7 no sirius package: cluster TIFF datasets cannot be opened");
                say("Cluster: the sirius package is missing on " + p.host + " (build it into the venv: " + pip + " install " + p.checkout + ")");
            } else {
                stepState(Step::Checks, StepStatus::Done, detail + " \xC2\xB7 sirius");
            }
        }

        void submit(const Profile& p) {
            stepState(Step::Submit, StepStatus::Running, "sbatch");
            token = ssh::randomHex(16);
            // Everything of the job's lives in ~/.sirius/run, a directory
            // only this user can enter, and every file in it is created
            // 0600 (umask 077, which sbatch hands on to the job's log too).
            std::string script = "umask 077\n";
            script += "mkdir -p \"$HOME/.sirius/run\" && chmod 700 \"$HOME/.sirius/run\" || exit 4\n";
            // token files of jobs that never started, a day old
            script += "find \"$HOME/.sirius/run\" -maxdepth 1 -name 'token.*' -mmin +1440 -exec rm -f {} + 2>/dev/null\n";
            script += "cd " + remotePathWord(p.checkout) + " || exit 3\n";
            // The token is written to a private file over this command
            // channel (printf is a builtin: no argument list shows it) and the
            // job is told only the file's name: in the job's environment
            // Slurm's accounting could keep it (AccountingStoreFlags=job_env).
            // The worker reads the file and deletes it.
            script += "tf=$(mktemp \"$HOME/.sirius/run/token.XXXXXXXX\") || exit 4\n";
            script += "printf '%s' " + shellQuote(token) + " > \"$tf\" || { rm -f \"$tf\"; exit 4; }\n";
            script += "unset SIRIUS_TOKEN\nexport SIRIUS_TOKEN_FILE=\"$tf\"\n";
            // port 0: the worker takes a free one and says which in its log
            script += "export SIRIUS_PORT=0\nexport SIRIUS_MAX_CLIENTS=8\n";
            if (!p.container.empty()) {
                // the job runs the worker in the image (sirius_worker.sbatch); no venv
                script += "unset SIRIUS_VENV\nexport SIRIUS_CONTAINER=" + remotePathWord(p.container) + "\n";
                script += "export SIRIUS_LAUNCHER=" + shellQuote(p.launcher.empty() ? std::string("apptainer") : p.launcher) + "\n";
                // the binds and the extra PYTHONPATH, quoted (a "~/" bind made $HOME): never the token
                if (const std::string bw = bindWord(p.bind); !bw.empty()) script += "export SIRIUS_CONTAINER_BIND=" + bw + "\n";
                else script += "unset SIRIUS_CONTAINER_BIND\n";
                if (const std::string pp = trim(p.containerPythonPath); !pp.empty()) script += "export SIRIUS_CONTAINER_PYTHONPATH=" + shellQuote(pp) + "\n";
                else script += "unset SIRIUS_CONTAINER_PYTHONPATH\n";
            } else if (!p.venv.empty()) {
                script += "export SIRIUS_VENV=" + remotePathWord(p.venv) + "\n";
            }
            if (p.gpus <= 0) script += "export SIRIUS_DEVICE=cpu\n";
            std::string cmd = "sbatch --parsable --job-name=sirius-worker --output=\"$HOME/.sirius/run/sirius-worker-%j.log\"";
            if (!p.partition.empty()) cmd += " --partition=" + shellQuote(p.partition);
            if (!p.account.empty()) cmd += " --account=" + shellQuote(p.account);
            if (!p.qos.empty()) cmd += " --qos=" + shellQuote(p.qos);
            if (!p.time.empty()) cmd += " --time=" + shellQuote(p.time);
            cmd += p.gpus > 0 ? " --gres=gpu:" + std::to_string(p.gpus) : std::string(" --gres=none");
            if (p.cpus > 0) cmd += " --cpus-per-task=" + std::to_string(p.cpus);
            if (!p.mem.empty()) cmd += " --mem=" + shellQuote(p.mem);
            script += "out=$(" + cmd + " app/python/slurm/sirius_worker.sbatch) || { rc=$?; rm -f \"$tf\"; printf '%s\\n' \"$out\"; exit $rc; }\n";
            script += "echo \"job=$out\"\n";
            const ssh::CommandResult r = remote(script, std::chrono::seconds(60));
            std::string id;
            const std::string job = keyValues(r.out)["job"];   // "4711" or "4711;cluster"
            for (char c : job) {
                if (std::isdigit(static_cast<unsigned char>(c))) id.push_back(c);
                else if (!id.empty()) break;
            }
            if (!r.ok() || id.empty())
                throw Failure{Step::Submit, "sbatch refused the worker job.", trim(r.err.empty() ? r.out : r.err),
                              "Check the partition, account, QoS and time in the profile."};
            update([&](Status& x) { x.jobId = id; });
            std::string where = p.partition.empty() ? std::string() : " to " + p.partition;
            stepState(Step::Submit, StepStatus::Done, "job " + id + where);
            say("Cluster: submitted job " + id + where);
        }

        // The job's log, in the private ~/.sirius/run (submit).
        static std::string jobLogPath(const std::string& id) { return "\"$HOME/.sirius/run/sirius-worker-" + id + ".log\""; }

        std::string jobLog(const Profile&, const std::string& id) {
            try {
                return trim(remote("tail -n 30 " + jobLogPath(id) + " 2>/dev/null", std::chrono::seconds(30)).out);
            } catch (const ssh::SshError&) {
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

        std::string waitInQueue(const Profile& p, const std::string& id) {
            stepState(Step::Queue, StepStatus::Running, "job " + id);
            const auto t0 = std::chrono::steady_clock::now();
            for (;;) {
                const ssh::CommandResult r = remote("squeue -h -j " + id + " -o '%T|%r|%N' 2>/dev/null", std::chrono::seconds(30));
                const std::string line = trim(r.out);
                if (line.empty()) {
                    const std::string fin = finalState(id);
                    throw Failure{Step::Queue, "Job " + id + " ended before the worker ran: " + fin + ".", jobLog(p, id), {}};
                }
                std::string state = line, reason, node;
                std::size_t a = line.find('|');
                if (a != std::string::npos) {
                    state = line.substr(0, a);
                    const std::size_t b = line.find('|', a + 1);
                    reason = line.substr(a + 1, b == std::string::npos ? std::string::npos : b - a - 1);
                    if (b != std::string::npos) node = line.substr(b + 1);
                }
                update([&](Status& x) { x.jobState = state; });
                if (state == "RUNNING" && !node.empty() && node != "(null)") {
                    stepState(Step::Queue, StepStatus::Done, "job " + id + " \xC2\xB7 waited " + elapsed(std::chrono::steady_clock::now() - t0));
                    return node;
                }
                std::string detail = "job " + id + " \xC2\xB7 " + state;
                if (!reason.empty() && reason != "None") detail += " (" + reason + ")";
                detail += " \xC2\xB7 " + elapsed(std::chrono::steady_clock::now() - t0);
                stepState(Step::Queue, StepStatus::Running, detail);
                if (!sleepCancellable(queuePoll)) throw Failure{Step::Queue, "Cancelled while job " + id + " waits in the queue.", {}, {}};
            }
        }

        void waitForWorker(const Profile& p, const std::string& id, const std::string& node) {
            stepState(Step::Start, StepStatus::Running, "on " + node);
            update([&](Status& x) { x.node = node; });
            say("Cluster: job " + id + " runs on " + node);
            const auto t0 = std::chrono::steady_clock::now();
            const std::string logFile = jobLogPath(id);
            for (;;) {
                const std::string script = "echo state=$(squeue -h -j " + id + " -o %T 2>/dev/null)\n[ -f " + logFile + " ] && grep -m1 -E '^\\{\"(port|error)\"' " +
                                           logFile + " | sed 's/^/announce=/'\ntrue\n";
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
                        std::string missing;
                        if (j.contains("missing") && j["missing"].is_array())
                            for (const json& x : j["missing"]) missing += (missing.empty() ? "" : ", ") + x.get<std::string>();
                        throw Failure{Step::Start, "The worker on " + node + " cannot start: " + (missing.empty() ? std::string("a package is missing") : missing + " missing") + ".",
                                      jobLog(p, id),
                                      !p.container.empty() ? "Rebuild the container image " + p.container + " with what is missing, or ask whoever builds it for the current one."
                                                           : (p.venv.empty() ? std::string("python3 -m pip") : p.venv + "/bin/pip") + " install -r " + p.checkout + "/app/python/requirements.txt"};
                    }
                    if (j.contains("port") && j["port"].is_number_integer()) {
                        // the worker took a free port (--port 0) and says which
                        const int port = j["port"].get<int>();
                        if (port <= 0 || port > 65535)
                            throw Failure{Step::Start, "The worker on " + node + " announced port " + std::to_string(port) + ".", jobLog(p, id), {}};
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
                    throw Failure{Step::Start, "Job " + id + " ended before the worker listened: " + fin + ".", jobLog(p, id), {}};
                }
                stepState(Step::Start, StepStatus::Running, "on " + node + " \xC2\xB7 starting \xC2\xB7 " + elapsed(std::chrono::steady_clock::now() - t0));
                if (!sleepCancellable(queuePoll)) throw Failure{Step::Start, "Cancelled while the worker starts.", {}, {}};
            }
        }

        void hello(const Profile&, const std::string& node) {
            int port = 0;
            {
                const std::lock_guard<std::mutex> g(m);
                port = workerPort;
            }
            stepState(Step::Hello, StepStatus::Running, node + ":" + std::to_string(port) + " through the SSH tunnel");
            int socks = 0;
            {
                auto s = sshSession();
                socks = s ? s->socksPort() : 0;
            }
            std::unique_ptr<RemoteWorker> w;
            try {
                w = RemoteWorker::connect(node, port, token, std::chrono::seconds(20), [this] { return cancel.load(); }, socks);
            } catch (const CancelledError&) {
                throw Failure{Step::Hello, "Cancelled while the worker answers.", {}, {}};
            } catch (const std::exception& e) {
                throw Failure{Step::Hello, "Could not reach the worker on " + node + ":" + std::to_string(port) + " through the SSH tunnel.", e.what(), {}};
            }
            w->setCancelGrace(std::chrono::milliseconds(0));
            const WorkerCapabilities caps = w->capabilities();
            int kinds = 0;
            for (const std::string& meth : caps.methods)
                if (meth.rfind("run:", 0) == 0) ++kinds;
            {
                const std::lock_guard<std::mutex> g(controlMutex);
                control = std::move(w);
            }
            update([&](Status& x) { x.caps = caps; });
            stepState(Step::Hello, StepStatus::Done,
                      "sirius_worker " + caps.version + " \xC2\xB7 " + caps.device + " \xC2\xB7 " + std::to_string(kinds) + " step kinds");
        }

        void run() {
            Profile p;
            bool submitting = true;
            {
                const std::lock_guard<std::mutex> g(m);
                p = profile;
                submitting = submitJob;
            }
            try {
                auto s = sshSession();
                const bool reuse = s && s->isOpen() && s->host() == p.host;
                if (reuse) stepState(Step::Login, StepStatus::Done, "logged in to " + p.host);
                else {
                    if (s) s->close();
                    login(p);
                }
                listPartitions(p, !submitting);
                if (!submitting) {
                    // logged in, the partitions listed: nothing is submitted
                    setState(State::Idle);
                    connecting.store(false);
                    return;
                }
                checks(p);
                submit(p);
                std::string id;
                {
                    const std::lock_guard<std::mutex> g(m);
                    id = status.jobId;
                }
                const std::string node = waitInQueue(p, id);
                waitForWorker(p, id, node);
                hello(p, node);
                setState(State::Connected);
                Status st;
                {
                    const std::lock_guard<std::mutex> g(m);
                    st = status;
                }
                say("HPC: connected to the worker on " + st.node + " (job " + st.jobId + ", " + st.caps.device + ")");
                startKeeper();
            } catch (const Failure& f) {
                update([&](Status& x) {
                    x.steps[static_cast<std::size_t>(f.step)] = StepState{StepStatus::Failed, f.reason};
                    x.state = State::Disconnected;
                    x.since = std::chrono::steady_clock::now();
                    x.reason = f.reason;
                    x.remoteOutput = f.remote;
                    if (!f.fix.empty()) x.fix = f.fix;
                    auto s2 = ssh;
                    x.sshUp = s2 && s2->isOpen();
                });
                say("HPC: " + f.reason + (f.remote.empty() ? std::string() : " \xE2\x80\x94 " + f.remote.substr(0, 300)));
            } catch (const std::exception& e) {
                update([&](Status& x) {
                    x.state = State::Disconnected;
                    x.since = std::chrono::steady_clock::now();
                    x.reason = cancel.load() ? std::string("Cancelled.") : std::string(e.what());
                    for (StepState& st : x.steps)
                        if (st.status == StepStatus::Running) st = StepState{StepStatus::Failed, x.reason};
                    auto s2 = ssh;
                    x.sshUp = s2 && s2->isOpen();
                });
                say(std::string("HPC: ") + (cancel.load() ? "cancelled" : e.what()));
            }
            connecting.store(false);
        }

        // --- keep-alive ------------------------------------------------------------------
        void startKeeper() {
            stopKeeper.store(false);
            if (keeper.joinable()) keeper.join();
            keeper = std::thread([this] { keep(); });
        }

        void lost(const std::string& reason, const std::string& remoteText = {}) {
            {
                const std::lock_guard<std::mutex> g(controlMutex);
                control.reset();
            }
            update([&](Status& x) {
                x.state = State::Disconnected;
                x.since = std::chrono::steady_clock::now();
                x.reason = reason;
                x.remoteOutput = remoteText;
                auto s = ssh;
                x.sshUp = s && s->isOpen();
            });
            say("HPC: disconnected: " + reason);
        }

        void keep() {
            int tick = 0;
            for (;;) {
                {
                    std::unique_lock<std::mutex> lk(keeperMutex);
                    if (keeperWake.wait_for(lk, keepAlive, [this] { return stopKeeper.load(); })) return;
                }
                ++tick;
                Profile p;
                std::string id, host;
                {
                    const std::lock_guard<std::mutex> g(m);
                    p = profile;
                    id = status.jobId;
                    host = status.host;
                }
                auto s = sshSession();
                if (!s || !s->isOpen()) {
                    lost("the SSH connection to " + host + " ended", s ? s->stderrTail() : std::string());
                    return;
                }
                bool pingFailed = false;
                std::string why;
                {
                    const std::lock_guard<std::mutex> g(controlMutex);
                    if (!control) return;
                    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
                    try {
                        control->call("ping", json::object(), {}, {}, [&] { return stopKeeper.load() || std::chrono::steady_clock::now() > deadline; });
                    } catch (const std::exception& e) {
                        if (stopKeeper.load()) return;
                        pingFailed = true;
                        why = isCancellation(e) ? std::string("no answer within 20 s") : std::string(e.what());
                    }
                }
                if (pingFailed || tick % 2 == 0) {
                    std::string state;
                    try {
                        state = trim(remote("squeue -h -j " + id + " -o %T 2>/dev/null", std::chrono::seconds(30)).out);
                    } catch (const std::exception&) {
                        state = "?";
                    }
                    if (state.empty() || (state != "RUNNING" && state != "COMPLETING" && state != "?")) {
                        lost("job " + id + " ended: " + finalState(id), jobLog(p, id));
                        return;
                    }
                    if (pingFailed) {
                        lost("the worker stopped answering (" + why + ")", jobLog(p, id));
                        return;
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
    };

    Session::Session() : impl_(std::make_unique<Impl>()) {}

    Session::~Session() {
        impl_->stopKeeperThread();
        impl_->stopWorkerThread();
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

    void Session::connect(const Profile& profile) { start(profile, true); }

    void Session::logIn(const Profile& profile) { start(profile, false); }

    void Session::start(const Profile& profile, bool submit) {
        if (impl_->connecting.exchange(true)) return;
        impl_->stopKeeperThread();
        if (impl_->worker.joinable()) impl_->worker.join();
        {
            const std::lock_guard<std::mutex> g(impl_->controlMutex);
            impl_->control.reset();
        }
        impl_->cancel.store(false);
        impl_->update([&](Status& x) {
            const bool sshUp = x.sshUp;
            const std::string host = x.host;
            x = Status{};
            x.state = State::Connecting;
            x.since = std::chrono::steady_clock::now();
            x.sshUp = sshUp;
            x.host = host;
        });
        {
            const std::lock_guard<std::mutex> g(impl_->m);
            impl_->profile = profile;
            impl_->workerPort = 0;
            impl_->submitJob = submit;
        }
        impl_->worker = std::thread([this] { impl_->run(); });
    }

    void Session::cancelConnect() {
        impl_->cancel.store(true);
        impl_->abortLogin.store(true);
    }

    void Session::disconnect(bool cancelJob) {
        impl_->stopKeeperThread();
        impl_->stopWorkerThread();
        {
            const std::lock_guard<std::mutex> g(impl_->controlMutex);
            impl_->control.reset();
        }
        std::string id, note;
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
