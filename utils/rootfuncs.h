#include <filesystem>

float calcWeights(float mc_weight, float lumi_sample, float sumw) {
	return (mc_weight * lumi_sample) / sumw;
};

float calcTruthWeights(float weight, float KFactor, float pileup) {
	return weight * KFactor * pileup;
};

bool filterPeak(float inv_mass) {
	return inv_mass < 120.;
};

float getVecVal(ROOT::VecOps::RVec<float> x, int i = 0);

float getVecVal(ROOT::VecOps::RVec<float> x, int i) {
    if (x.size() > i)  return x[i];
    else               return NAN;
};

// extract sum of weights from DTA outputs
std::map<int, float> get_sumw(std::string data_dir, std::string ttree_name) {
    std::map<int, float> dsid_map;
    std::cout << "data dir: " << data_dir << std::endl;

    // get vector of ROOT files
    std::filesystem::path data_path(data_dir);
    if (!std::filesystem::is_directory(data_dir)) {
        throw std::runtime_error(data_path.string() + " is not a folder");
    }
    std::vector<std::string> paths_vec;
    for (const auto& entry : std::filesystem::directory_iterator(data_dir)) {
        const auto full_path = entry.path().string();

        if (entry.is_regular_file()) {
            const auto filename = entry.path().filename().string();
            std::cout << "Found file " << filename << std::endl;
            if (filename.substr(filename.find_last_of(".") + 1) == "root")
                paths_vec.push_back(full_path);
        }
    }

    // calculate sum of weights for each dataset ID
    for (auto filepath: paths_vec) {
        std::cout << "file: " << filepath << std::endl;
        TFile file(filepath.c_str(), "READ");

        TTree* tree = (TTree*)file.Get(ttree_name.c_str());
        tree->GetEntry(0);
        UInt_t dsid;
        tree->SetBranchAddress("mcChannel", &dsid);

        float sumw = ((TH1F*)file.Get("sumOfWeights"))->GetBinContent(4);
        if (!(dsid_map.find(dsid) == dsid_map.end()))
            dsid_map[dsid] += sumw;
        else
            dsid_map[dsid] = sumw;
    }

    return dsid_map;
};
