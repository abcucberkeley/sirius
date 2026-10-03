#include "core/cluster_profiles.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>

#include "core/settings_toml.hpp"

namespace sirius::app::cluster {

    using json = nlohmann::json;

    namespace {

        std::string trim(const std::string& s) {
            const std::size_t a = s.find_first_not_of(" \t\r\n");
            if (a == std::string::npos) return {};
            const std::size_t b = s.find_last_not_of(" \t\r\n");
            return s.substr(a, b - a + 1);
        }

        // "a, b:/b:ro,,c" -> {"a", "b:/b:ro", "c"}
        std::vector<std::string> bindEntries(const std::string& bind) {
            std::vector<std::string> out;
            std::string cur;
            for (const char c : bind + ",") {
                if (c == ',') {
                    if (!trim(cur).empty()) out.push_back(trim(cur));
                    cur.clear();
                } else {
                    cur.push_back(c);
                }
            }
            return out;
        }

        std::string joinBinds(const std::vector<std::string>& entries) {
            std::string out;
            for (const std::string& e : entries)
                if (!trim(e).empty()) out += (out.empty() ? "" : ",") + trim(e);
            return out;
        }

        std::vector<std::string> strings(const json& j) {
            std::vector<std::string> out;
            if (j.is_array())
                for (const json& e : j)
                    if (e.is_string()) out.push_back(e.get<std::string>());
            return out;
        }

        void str(const json& j, const char* k, std::string& out) {
            if (j.contains(k) && j[k].is_string()) out = j[k].get<std::string>();
        }

        void num(const json& j, const char* k, int& out) {
            if (j.contains(k) && j[k].is_number_integer()) out = j[k].get<int>();
        }

        std::string memText(long long mb) { return mb >= 1024 ? std::to_string(mb / 1024) + "G" : std::to_string(mb) + "M"; }

    } // namespace

    // --- the profile ---------------------------------------------------------------------

    std::string Profile::displayName() const {
        if (!trim(name).empty()) return trim(name);
        if (!trim(host).empty()) return trim(host);
        return "New cluster";
    }

    const PartitionChoice* Profile::choice(const std::string& part) const {
        for (const PartitionChoice& c : choices)
            if (c.name == part) return &c;
        return nullptr;
    }

    json toJson(const PartitionChoice& c) {
        json j = {{"name", c.name}, {"default", c.isDefault}, {"accounts", c.accounts}, {"qos", c.qos}, {"max_time", c.maxTime}};
        if (!c.times.empty()) j["times"] = c.times;
        if (c.gpus >= 0) j["gpus"] = c.gpus;
        if (c.cpus >= 0) j["cpus"] = c.cpus;
        if (!c.mem.empty()) j["mem"] = c.mem;
        if (c.maxGpus >= 0) j["max_gpus"] = c.maxGpus;
        if (c.maxCpus >= 0) j["max_cpus"] = c.maxCpus;
        if (!c.maxMem.empty()) j["max_mem"] = c.maxMem;
        return j;
    }

    PartitionChoice partitionChoiceFromJson(const json& j) {
        PartitionChoice c;
        if (!j.is_object()) return c;
        str(j, "name", c.name);
        if (j.contains("default") && j["default"].is_boolean()) c.isDefault = j["default"].get<bool>();
        if (j.contains("accounts")) c.accounts = j["accounts"].is_string() ? std::vector<std::string>{j["accounts"].get<std::string>()} : strings(j["accounts"]);
        if (j.contains("qos")) c.qos = j["qos"].is_string() ? std::vector<std::string>{j["qos"].get<std::string>()} : strings(j["qos"]);
        str(j, "max_time", c.maxTime);
        c.times = strings(j.value("times", json::array()));
        num(j, "gpus", c.gpus);
        num(j, "cpus", c.cpus);
        str(j, "mem", c.mem);
        num(j, "max_gpus", c.maxGpus);
        num(j, "max_cpus", c.maxCpus);
        str(j, "max_mem", c.maxMem);
        return c;
    }

    json Profile::toJson() const {
        json bindSetsJson = json::array();
        for (const std::string& b : bindSets) bindSetsJson.push_back(bindEntries(b));
        json parts = json::array();
        for (const PartitionChoice& c : choices) parts.push_back(cluster::toJson(c));
        json j = {{"host", host},
                  {"image", container},
                  {"images", images},
                  {"binds", bindEntries(bind)},
                  {"bind_sets", bindSetsJson},
                  {"checkout", checkout},
                  {"launcher", launcher},
                  {"python_path", containerPythonPath},
                  {"engine", engine},
                  {"engine_builds", engineBuilds},
                  {"engine_bin", engineBin},
                  {"cache", scratch},
                  {"def_file", defFile},
                  {"job", {{"partition", partition}, {"account", account}, {"qos", qos}, {"time", time}, {"gpus", gpus}, {"cpus", cpus}, {"mem", mem}}},
                  {"partitions", parts}};
        if (!sshProgram.empty()) j["ssh"] = sshProgram;
        if (!perHost.empty()) {
            json hosts = json::object();
            for (const auto& [h, c] : perHost) {
                json one = {{"partition", c.partition}, {"account", c.account}, {"qos", c.qos}, {"time", c.time}};
                if (c.bind) one["binds"] = bindEntries(*c.bind);
                if (c.containerPythonPath) one["python_path"] = *c.containerPythonPath;
                hosts[h] = one;
            }
            j["last_used"] = hosts;
        }
        return j;
    }

    Profile Profile::fromJson(const json& j) {
        Profile p;
        if (!j.is_object()) return p;
        str(j, "name", p.name);
        str(j, "host", p.host);
        str(j, "checkout", p.checkout);
        // the image: "image", or "container" in the JSON settings of before
        str(j, "container", p.container);
        str(j, "image", p.container);
        p.images = strings(j.value("images", json::array()));
        str(j, "launcher", p.launcher);
        if (p.launcher.empty()) p.launcher = "apptainer";
        str(j, "bind", p.bind);
        if (j.contains("binds")) p.bind = j["binds"].is_string() ? j["binds"].get<std::string>() : joinBinds(strings(j["binds"]));
        if (j.contains("bind_sets") && j["bind_sets"].is_array())
            for (const json& set : j["bind_sets"]) {
                const std::string b = set.is_string() ? set.get<std::string>() : joinBinds(strings(set));
                if (!b.empty()) p.bindSets.push_back(b);
            }
        str(j, "containerPythonPath", p.containerPythonPath);
        str(j, "python_path", p.containerPythonPath);
        // the job: a [job] table, or flat keys in the JSON settings of before
        const json& job = j.contains("job") && j["job"].is_object() ? j["job"] : j;
        str(job, "partition", p.partition);
        str(job, "account", p.account);
        str(job, "qos", p.qos);
        str(job, "time", p.time);
        num(job, "gpus", p.gpus);
        num(job, "cpus", p.cpus);
        str(job, "mem", p.mem);
        num(j, "port", p.port);
        str(j, "ssh", p.sshProgram);
        if (j.contains("engine") && j["engine"].is_boolean()) p.engine = j["engine"].get<bool>();
        str(j, "engineBin", p.engineBin);
        str(j, "engine_bin", p.engineBin);
        str(j, "engine_builds", p.engineBuilds);
        str(j, "scratch", p.scratch);
        str(j, "cache", p.scratch);
        str(j, "def_file", p.defFile);
        if (j.contains("partitions") && j["partitions"].is_array())
            for (const json& c : j["partitions"]) {
                PartitionChoice pc = partitionChoiceFromJson(c);
                if (!trim(pc.name).empty()) p.choices.push_back(pc);
            }
        const char* hostsKey = j.contains("last_used") ? "last_used" : "perHost";
        if (j.contains(hostsKey) && j[hostsKey].is_object())
            for (const auto& [h, c] : j[hostsKey].items()) {
                if (!c.is_object()) continue;
                SlurmChoice s;
                str(c, "partition", s.partition);
                str(c, "account", s.account);
                str(c, "qos", s.qos);
                str(c, "time", s.time);
                if (c.contains("bind") && c["bind"].is_string()) s.bind = c["bind"].get<std::string>();
                if (c.contains("binds")) s.bind = c["binds"].is_string() ? c["binds"].get<std::string>() : joinBinds(strings(c["binds"]));
                if (c.contains("containerPythonPath") && c["containerPythonPath"].is_string())
                    s.containerPythonPath = c["containerPythonPath"].get<std::string>();
                if (c.contains("python_path") && c["python_path"].is_string()) s.containerPythonPath = c["python_path"].get<std::string>();
                p.perHost[h] = s;
            }
        return p;
    }

    ProfileChange profileChange(const Profile& a, const Profile& b) {
        ProfileChange c;
        auto job = [&](const std::string& what, bool differs) {
            if (!differs) return;
            c.newJob = true;
            c.fields.push_back(what);
        };
        auto worker = [&](const std::string& what, bool differs) {
            if (!differs) return;
            c.newWorker = true;
            c.fields.push_back(what);
        };
        job("cluster", trim(a.host) != trim(b.host));
        job("partition", trim(a.partition) != trim(b.partition));
        job("account", trim(a.account) != trim(b.account));
        job("QoS", trim(a.qos) != trim(b.qos));
        job("time limit", trim(a.time) != trim(b.time));
        job("GPUs", a.gpus != b.gpus);
        job("CPUs", a.cpus != b.cpus);
        job("memory", trim(a.mem) != trim(b.mem));
        worker("worker image", trim(a.container) != trim(b.container));
        worker("data folders", joinBinds(bindEntries(a.bind)) != joinBinds(bindEntries(b.bind)));
        worker("SIRIUS checkout", trim(a.checkout) != trim(b.checkout));
        worker("container launcher", trim(a.launcher) != trim(b.launcher));
        worker("extra Python path", trim(a.containerPythonPath) != trim(b.containerPythonPath));
        worker("C++ engine", a.engine != b.engine);
        worker("engine builds", trim(a.engineBuilds) != trim(b.engineBuilds));
        worker("engine executable", trim(a.engineBin) != trim(b.engineBin));
        worker("node cache folder", trim(a.scratch) != trim(b.scratch));
        return c;
    }

    // --- the profiles --------------------------------------------------------------------

    std::string profileKey(const std::string& name) { return "cluster/" + name; }

    bool isReservedName(const std::string& name) { return name == "current" || name == "recentFolders" || name == "profile"; }

    ProfileBook ProfileBook::fromSettings(const json& flat, bool* migrated) {
        ProfileBook b;
        if (migrated) *migrated = false;
        if (!flat.is_object()) return b;
        for (auto it = flat.begin(); it != flat.end(); ++it) {
            const std::string& key = it.key();
            if (key.rfind("cluster/", 0) != 0 || !it.value().is_object()) continue;
            const std::string name = key.substr(8);
            if (name.empty() || isReservedName(name)) continue;
            Profile p = Profile::fromJson(it.value());
            p.name = name;
            b.profiles.push_back(std::move(p));
        }
        if (const auto c = flat.find(kCurrentKey); c != flat.end() && c->is_string()) b.current = c->get<std::string>();
        if (b.profiles.empty())
            if (const auto legacy = flat.find(kLegacyProfileKey); legacy != flat.end() && legacy->is_object()) {
                // the one profile of before: named after its host, every value
                // kept, its partition, account and QoS the dropdowns' first choice
                Profile p = Profile::fromJson(*legacy);
                p.name = b.uniqueName(trim(p.host).empty() ? std::string("My cluster") : p.host);
                if (!trim(p.partition).empty() && !p.choice(p.partition)) {
                    PartitionChoice c;
                    c.name = trim(p.partition);
                    c.isDefault = true;
                    if (!trim(p.account).empty()) c.accounts.push_back(trim(p.account));
                    if (!trim(p.qos).empty()) c.qos.push_back(trim(p.qos));
                    p.choices.push_back(c);
                }
                if (!trim(p.container).empty() && std::find(p.images.begin(), p.images.end(), trim(p.container)) == p.images.end())
                    p.images.push_back(trim(p.container));
                b.current = p.name;
                b.profiles.push_back(std::move(p));
                if (migrated) *migrated = true;
            }
        if (!b.find(b.current)) b.current = b.profiles.empty() ? std::string() : b.profiles.front().name;
        return b;
    }

    std::map<std::string, json> ProfileBook::toSettings() const {
        std::map<std::string, json> out;
        for (const Profile& p : profiles) out[profileKey(p.name)] = p.toJson();
        out[kCurrentKey] = current;
        return out;
    }

    std::vector<std::string> ProfileBook::staleKeys(const json& flat) const {
        std::vector<std::string> out;
        if (!flat.is_object()) return out;
        for (auto it = flat.begin(); it != flat.end(); ++it) {
            const std::string& key = it.key();
            if (key.rfind("cluster/", 0) != 0 || !it.value().is_object()) continue;
            const std::string name = key.substr(8);
            if (name.empty() || isReservedName(name)) continue;
            if (!find(name)) out.push_back(key);
        }
        return out;
    }

    Profile* ProfileBook::find(const std::string& n) {
        for (Profile& p : profiles)
            if (p.name == n) return &p;
        return nullptr;
    }

    const Profile* ProfileBook::find(const std::string& n) const {
        for (const Profile& p : profiles)
            if (p.name == n) return &p;
        return nullptr;
    }

    Profile ProfileBook::currentProfile() const {
        const Profile* p = find(current);
        return p ? *p : Profile{};
    }

    std::string ProfileBook::nameProblem(const std::string& name, const std::string& except) const {
        const std::string n = trim(name);
        if (n.empty()) return "A profile needs a name.";
        if (n.find('/') != std::string::npos) return "A profile's name cannot hold a slash.";
        if (isReservedName(n)) return "\"" + n + "\" is a name the settings use for themselves: choose another.";
        if (n != except && find(n)) return "There is a profile called \"" + n + "\" already.";
        return {};
    }

    std::string ProfileBook::uniqueName(const std::string& base) const {
        std::string b = trim(base);
        std::replace(b.begin(), b.end(), '/', '-');
        if (b.empty()) b = "New cluster";
        if (nameProblem(b).empty()) return b;
        for (int i = 2;; ++i) {
            const std::string n = b + " " + std::to_string(i);
            if (nameProblem(n).empty()) return n;
        }
    }

    void ProfileBook::put(Profile p) {
        p.name = p.displayName();
        if (Profile* there = find(p.name)) {
            *there = p;
        } else {
            profiles.push_back(p);
            std::sort(profiles.begin(), profiles.end(), [](const Profile& a, const Profile& b) { return a.name < b.name; });
        }
        current = p.name;
    }

    bool ProfileBook::rename(const std::string& from, const std::string& to) {
        const std::string t = trim(to);
        Profile* p = find(from);
        if (!p) return false;
        if (t == from) return true;
        if (!nameProblem(t, from).empty()) return false;
        p->name = t;
        if (current == from) current = t;
        std::sort(profiles.begin(), profiles.end(), [](const Profile& a, const Profile& b) { return a.name < b.name; });
        return true;
    }

    bool ProfileBook::remove(const std::string& n) {
        const auto it = std::find_if(profiles.begin(), profiles.end(), [&](const Profile& p) { return p.name == n; });
        if (it == profiles.end()) return false;
        profiles.erase(it);
        if (current == n || !find(current)) current = profiles.empty() ? std::string() : profiles.front().name;
        return true;
    }

    // --- the dropdowns -----------------------------------------------------------

    PartitionChoice choiceFromCluster(const ClusterInfo& info, const std::string& partition) {
        PartitionChoice c;
        c.name = partition;
        c.accounts = accountsFor(info, partition);
        std::set<std::string> seen;
        for (const std::string& a : c.accounts)
            for (const std::string& q : qosFor(info, partition, a))
                if (seen.insert(q).second) c.qos.push_back(q);
        if (const Partition* part = findPartition(info, partition)) {
            c.isDefault = part->isDefault;
            const long long t = slurmTimeSeconds(part->maxTime);
            if (t >= 0) c.maxTime = slurmTimeText(t);
            if (part->gpusPerNode > 0) c.maxGpus = part->gpusPerNode;
            else c.gpus = 0;   // a partition without GPUs: picking it asks for none
            if (part->cpusPerNode > 0) c.maxCpus = part->cpusPerNode;
            if (part->memPerNodeMB > 0) c.maxMem = memText(part->memPerNodeMB);
        }
        return c;
    }

    std::vector<std::string> applyChoice(Profile& p, const PartitionChoice& c) {
        std::vector<std::string> changed;
        p.partition = c.name;
        if (!c.accounts.empty() && std::find(c.accounts.begin(), c.accounts.end(), trim(p.account)) == c.accounts.end()) {
            p.account = c.accounts.front();
            changed.push_back("account " + p.account);
        }
        if (!c.qos.empty() && std::find(c.qos.begin(), c.qos.end(), trim(p.qos)) == c.qos.end()) {
            p.qos = c.qos.front();
            changed.push_back("QoS " + p.qos);
        }
        if (c.gpus >= 0 && p.gpus != c.gpus) {
            p.gpus = c.gpus;
            changed.push_back(std::to_string(p.gpus) + (p.gpus == 1 ? " GPU" : " GPUs"));
        }
        if (c.cpus >= 0 && p.cpus != c.cpus) {
            p.cpus = c.cpus;
            changed.push_back(std::to_string(p.cpus) + " CPUs");
        }
        if (!c.mem.empty() && trim(p.mem) != c.mem) {
            p.mem = c.mem;
            changed.push_back("memory " + p.mem);
        }
        if (c.maxGpus >= 0 && p.gpus > c.maxGpus) {
            p.gpus = c.maxGpus;
            changed.push_back(std::to_string(p.gpus) + (p.gpus == 1 ? " GPU" : " GPUs") + " (a node's)");
        }
        if (c.maxCpus > 0 && p.cpus > c.maxCpus) {
            p.cpus = c.maxCpus;
            changed.push_back(std::to_string(p.cpus) + " CPUs (a node's)");
        }
        if (const long long most = memoryMB(c.maxMem); most > 0 && memoryMB(p.mem) > most) {
            p.mem = c.maxMem;
            changed.push_back("memory " + p.mem + " (a node's)");
        }
        const long long limit = slurmTimeSeconds(c.maxTime), want = slurmTimeSeconds(p.time);
        if (limit >= 0 && want >= 0 && want > limit) {
            p.time = slurmTimeText(limit);
            changed.push_back("time " + p.time + " (the most allowed)");
        }
        return changed;
    }

    std::vector<std::string> timeChoices(const PartitionChoice* c, const std::string& limit, const std::string& current) {
        std::vector<std::string> out;
        if (c && !c->times.empty()) {
            out = c->times;
        } else {
            long long most = slurmTimeSeconds(limit);
            if (c) {
                const long long own = slurmTimeSeconds(c->maxTime);
                if (own >= 0 && (most < 0 || own < most)) most = own;
            }
            for (const char* t : {"00:30:00", "01:00:00", "02:00:00", "04:00:00", "08:00:00", "12:00:00", "1-00:00:00", "2-00:00:00", "3-00:00:00",
                                  "7-00:00:00"})
                if (most < 0 || slurmTimeSeconds(t) <= most) out.emplace_back(t);
        }
        const std::string cur = trim(current);
        if (!cur.empty() && std::find(out.begin(), out.end(), cur) == out.end()) {
            out.push_back(cur);
            std::stable_sort(out.begin(), out.end(), [](const std::string& a, const std::string& b) { return slurmTimeSeconds(a) < slurmTimeSeconds(b); });
        }
        return out;
    }

    // --- sharing a profile ---------------------------------------------------------

    std::string exportProfile(const Profile& p) {
        json flat = json::object();
        flat[profileKey(p.displayName())] = p.toJson();
        return "# A SIRIUS cluster profile. Import it with Process > Connect to cluster... >\n"
               "# Import..., or paste it into your settings file (Preferences > Edit settings file...).\n"
               "# It holds no password and no token.\n" +
               settings_toml::toToml(flat, false);
    }

    std::vector<Profile> importProfiles(const std::string& text) {
        const settings_toml::ParseResult r = settings_toml::fromToml(text);
        if (!r.ok)
            throw std::runtime_error("this is not a TOML file" +
                                     (r.line > 0 ? " (line " + std::to_string(r.line) + ", column " + std::to_string(r.column) + ": " + r.error + ")"
                                                 : " (" + r.error + ")"));
        ProfileBook b = ProfileBook::fromSettings(r.flat);
        if (b.profiles.empty()) throw std::runtime_error("it holds no cluster profile (a [cluster.<name>] table with a host)");
        return b.profiles;
    }

    // --- the settings editor's checks --------------------------------------------------

    namespace {

        const std::set<std::string>& profileKeys() {
            static const std::set<std::string> keys = {"host", "image", "images", "binds", "bind_sets", "checkout", "launcher", "python_path", "engine",
                                                       "engine_builds", "engine_bin", "cache", "def_file", "job", "partitions", "ssh", "last_used"};
            return keys;
        }

        // the names of before, read all the same: what each is now
        const std::map<std::string, std::string>& oldKeys() {
            static const std::map<std::string, std::string> keys = {
                {"container", "image"}, {"bind", "binds"}, {"containerPythonPath", "python_path"}, {"engineBin", "engine_bin"}, {"scratch", "cache"}, {"perHost", "last_used"}, {"partition", "[job] partition"}, {"account", "[job] account"}, {"qos", "[job] qos"}, {"time", "[job] time"}, {"gpus", "[job] gpus"}, {"cpus", "[job] cpus"}, {"mem", "[job] mem"}, {"port", "nothing (the worker takes a free port)"}, {"name", "the table's name"}, {"venv", "nothing: the worker runs in the image"}};
            return keys;
        }

        std::string list(const std::set<std::string>& keys) {
            std::string out;
            for (const std::string& k : keys) out += (out.empty() ? "" : ", ") + k;
            return out;
        }

        struct Checker {
            std::vector<SettingsProblem> out;
            void error(std::vector<std::string> path, const std::string& what) { out.push_back({std::move(path), what, true, 0, 0}); }
            void warn(std::vector<std::string> path, const std::string& what) { out.push_back({std::move(path), what, false, 0, 0}); }

            std::vector<std::string> at(const std::vector<std::string>& base, const std::string& k) {
                std::vector<std::string> p = base;
                p.push_back(k);
                return p;
            }

            void string(const json& t, const std::vector<std::string>& base, const char* k, const std::string& what) {
                if (t.contains(k) && !t[k].is_string()) error(at(base, k), std::string(k) + " must be text in quotes: " + what + ".");
            }
            void strings(const json& t, const std::vector<std::string>& base, const char* k, const std::string& what) {
                if (!t.contains(k)) return;
                const json& v = t[k];
                bool ok = v.is_array();
                if (ok)
                    for (const json& e : v) ok = ok && e.is_string();
                if (!ok) error(at(base, k), std::string(k) + " must be a list of texts in quotes, [\"a\", \"b\"]: " + what + ".");
            }
            void integer(const json& t, const std::vector<std::string>& base, const char* k, const std::string& what) {
                if (!t.contains(k)) return;
                if (!t[k].is_number_integer() || t[k].get<long long>() < 0)
                    error(at(base, k), std::string(k) + " must be a whole number, 0 or more: " + what + ".");
            }
            void time(const json& t, const std::vector<std::string>& base, const char* k) {
                if (!t.contains(k) || !t[k].is_string()) return;
                const std::string v = trim(t[k].get<std::string>());
                if (!v.empty() && slurmTimeSeconds(v) == -2)
                    error(at(base, k), std::string(k) + " = \"" + v + "\" is not a time Slurm reads: write it as \"01:30:00\", \"90\" (minutes) or \"2-00:00:00\".");
            }
            void memory(const json& t, const std::vector<std::string>& base, const char* k) {
                if (!t.contains(k) || !t[k].is_string()) return;
                const std::string v = trim(t[k].get<std::string>());
                if (!v.empty() && memoryMB(v) < 0)
                    error(at(base, k), std::string(k) + " = \"" + v + "\" is not a size Slurm reads: write it as \"64G\", \"500M\" or \"1T\".");
            }

            void job(const json& j, const std::vector<std::string>& base) {
                static const std::set<std::string> keys = {"partition", "account", "qos", "time", "gpus", "cpus", "mem"};
                for (auto it = j.begin(); it != j.end(); ++it)
                    if (!keys.count(it.key())) error(at(base, it.key()), "[job] has no key \"" + it.key() + "\" (it has " + list(keys) + ").");
                string(j, base, "partition", "the partition the job runs in");
                string(j, base, "account", "the account it is charged to");
                string(j, base, "qos", "the quality of service");
                string(j, base, "time", "the time limit, \"01:00:00\"");
                time(j, base, "time");
                integer(j, base, "gpus", "the GPUs the job asks for");
                integer(j, base, "cpus", "the CPU cores it asks for");
                string(j, base, "mem", "the memory it asks for, \"64G\"");
                memory(j, base, "mem");
            }

            void partition(const json& p, const std::vector<std::string>& base) {
                static const std::set<std::string> keys = {"name", "default", "accounts", "qos", "max_time", "times",
                                                           "gpus", "cpus", "mem", "max_gpus", "max_cpus", "max_mem"};
                if (!p.is_object()) {
                    error(base, "Each [[...partitions]] entry must be a table with a name.");
                    return;
                }
                for (auto it = p.begin(); it != p.end(); ++it)
                    if (!keys.count(it.key())) error(at(base, it.key()), "A partition has no key \"" + it.key() + "\" (it has " + list(keys) + ").");
                if (!p.contains("name") || !p["name"].is_string() || trim(p["name"].get<std::string>()).empty())
                    error(base, "This partition has no name: add name = \"<the partition, as sinfo lists it>\".");
                if (p.contains("default") && !p["default"].is_boolean()) error(at(base, "default"), "default must be true or false.");
                for (const char* k : {"accounts", "qos", "times"}) {
                    if (p.contains(k) && p[k].is_string()) continue;   // one, written without brackets
                    strings(p, base, k, "the choices the dropdown offers");
                }
                if (p.contains("times") && p["times"].is_array())
                    for (std::size_t i = 0; i < p["times"].size(); ++i)
                        if (p["times"][i].is_string() && slurmTimeSeconds(p["times"][i].get<std::string>()) == -2)
                            error(at(at(base, "times"), std::to_string(i)),
                                  "\"" + p["times"][i].get<std::string>() + "\" is not a time Slurm reads: write it as \"01:30:00\" or \"2-00:00:00\".");
                string(p, base, "max_time", "the longest time a job there may run");
                time(p, base, "max_time");
                for (const char* k : {"gpus", "cpus", "max_gpus", "max_cpus"}) integer(p, base, k, "a count");
                for (const char* k : {"mem", "max_mem"}) {
                    string(p, base, k, "a size, \"64G\"");
                    memory(p, base, k);
                }
            }

            void profile(const std::string& name, const json& p) {
                const std::vector<std::string> base = {"cluster", name};
                if (!p.is_object()) {
                    error(base, "cluster." + name + " must be a table: write it as [cluster." + name + "] with its keys under it.");
                    return;
                }
                for (auto it = p.begin(); it != p.end(); ++it) {
                    if (profileKeys().count(it.key())) continue;
                    if (const auto old = oldKeys().find(it.key()); old != oldKeys().end())
                        warn(at(base, it.key()), "\"" + it.key() + "\" is an older name: it is read as " + old->second + ".");
                    else
                        error(at(base, it.key()), "A cluster profile has no key \"" + it.key() + "\" (it has " + list(profileKeys()) + ").");
                }
                if (!p.contains("host") || !p["host"].is_string() || trim(p["host"].get<std::string>()).empty())
                    error(at(base, "host"), "[cluster." + name + "] needs host = \"<the cluster's name in your ~/.ssh/config, or user@host>\".");
                if (!p.contains("image") && !p.contains("container"))
                    error(base, "[cluster." + name + "] needs image = \"<the worker image (.sif) on the cluster>\" (write image = \"\" while you have none).");
                else if (p.contains("image") && p["image"].is_string() && trim(p["image"].get<std::string>()).empty())
                    warn(at(base, "image"), "No worker image yet: the worker cannot start until image names one (Connect to cluster can build it).");
                for (const char* k : {"host", "image", "checkout", "launcher", "python_path", "engine_builds", "engine_bin", "cache", "def_file", "ssh"})
                    string(p, base, k, "a name or a path");
                strings(p, base, "images", "worker images used before");
                strings(p, base, "binds", "folders on the cluster the image sees");
                if (p.contains("bind_sets")) {
                    bool ok = p["bind_sets"].is_array();
                    if (ok)
                        for (const json& s : p["bind_sets"]) {
                            bool inner = s.is_string() || s.is_array();
                            if (s.is_array())
                                for (const json& e : s) inner = inner && e.is_string();
                            ok = ok && inner;
                        }
                    if (!ok) error(at(base, "bind_sets"), "bind_sets must be a list of folder lists, [[\"/data\"], [\"/data\", \"/scratch\"]].");
                }
                if (p.contains("engine") && !p["engine"].is_boolean()) error(at(base, "engine"), "engine must be true or false.");
                if (p.contains("job")) {
                    if (p["job"].is_object()) job(p["job"], at(base, "job"));
                    else error(at(base, "job"), "job must be a table: [cluster." + name + ".job] with partition, account, qos, time, gpus, cpus, mem.");
                }
                if (p.contains("partitions")) {
                    if (!p["partitions"].is_array()) {
                        error(at(base, "partitions"), "partitions must be a list of tables: [[cluster." + name + ".partitions]] for each.");
                    } else {
                        int defaults = 0;
                        for (std::size_t i = 0; i < p["partitions"].size(); ++i) {
                            const json& e = p["partitions"][i];
                            partition(e, at(at(base, "partitions"), std::to_string(i)));
                            if (e.is_object() && e.value("default", false) == true) ++defaults;
                        }
                        if (defaults > 1) warn(at(base, "partitions"), "More than one partition says default = true: the first is taken.");
                    }
                }
                if (p.contains("last_used") && !p["last_used"].is_object()) error(at(base, "last_used"), "last_used must be a table of hosts.");
            }
        };

    } // namespace

    std::vector<SettingsProblem> checkClusterSettings(const json& flat) {
        Checker c;
        if (!flat.is_object()) return c.out;
        std::set<std::string> names;
        for (auto it = flat.begin(); it != flat.end(); ++it) {
            const std::string& key = it.key();
            if (key.rfind("cluster/", 0) != 0) continue;
            const std::string name = key.substr(8);
            if (name == "current") {
                if (!it.value().is_string()) c.error({"cluster", "current"}, "current must be the name of a profile, in quotes.");
                continue;
            }
            if (name == "recentFolders") {
                c.strings(json{{"recentFolders", it.value()}}, {"cluster"}, "recentFolders", "folders on the cluster");
                continue;
            }
            if (name == "profile") {
                c.warn({"cluster", "profile"}, "[cluster.profile] is the single profile of an older SIRIUS: it is read only while no other profile exists.");
                continue;
            }
            if (!it.value().is_object()) {
                c.error({"cluster", name}, "\"" + name + "\" under [cluster] is neither a profile (a table) nor one of current, recentFolders.");
                continue;
            }
            names.insert(name);
            c.profile(name, it.value());
        }
        if (const auto cur = flat.find(kCurrentKey); cur != flat.end() && cur->is_string() && !cur->get<std::string>().empty() &&
                                                     !names.count(cur->get<std::string>()))
            c.warn({"cluster", "current"}, "current = \"" + cur->get<std::string>() + "\" names no profile here: the first one is used.");
        return c.out;
    }

    std::vector<SettingsProblem> checkSettingsText(const std::string& text) {
        const settings_toml::ParseResult r = settings_toml::fromToml(text);
        if (!r.ok) return {SettingsProblem{{}, "This is not valid TOML: " + r.error + ".", true, r.line, r.column}};
        std::vector<SettingsProblem> out = checkClusterSettings(r.flat);
        for (SettingsProblem& p : out)
            if (const auto pos = settings_toml::position(text, p.path)) {
                p.line = pos->first;
                p.column = pos->second;
            }
        return out;
    }

} // namespace sirius::app::cluster
