#ifndef root_funcs
#define root_funcs

#include <filesystem>
#include <cmath>

const double PI = 3.141592653589793238463;

float calcWeights(float mc_weight, float lumi_sample, float sumw) {
	return (mc_weight * lumi_sample) / sumw;
};

float calcTruthWeights(float weight, float KFactor, float pileup) {
	return weight * KFactor * pileup;
};

bool filterPeak(float inv_mass) {
	return inv_mass < 120.;
};

float getVecVal(ROOT::VecOps::RVec<float>* x, int i = 0);

float getVecVal(ROOT::VecOps::RVec<double>* x, int i = 0);

float getVecVal(ROOT::VecOps::RVec<double>* x, int i) {
    if (x->size() > i)  return x->at(i);
    else                return NAN;
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

bool isnan(ROOT::VecOps::RVec<float> var) {
	if (var.size() == 0) return true;
	else return false;
}


// Variable calculations
double mt(double l1_pt, double l2_pt, double l1_phi, double l2_phi) {
	double dphi = std::abs(l1_phi - l2_phi);
	if (dphi > PI) dphi = 2 * PI - dphi;
	return std::sqrt(2.0 * l1_pt * l2_pt * (1 - std::cos(dphi)));
};

double mt(ROOT::VecOps::RVec<float> l1_pt, float l2_pt, ROOT::VecOps::RVec<float> l1_phi, float l2_phi) {
	double dphi = std::abs(l1_phi[0] - l2_phi);
	if (dphi > PI) dphi = 2 * PI - dphi;
	return std::sqrt(2.0 * l1_pt[0] * l2_pt * (1 - std::cos(dphi)));
};

double vy(double x1, double x2) {
	return 0.5 * std::log(x1 / x2);
}

double delta_z0_sintheta(double z0, double eta) {
	return z0 * std::sin(2.0 * std::atan(std::exp(-eta)));
}

double dilep_m(double m1, double m2) {
	return m1 + m2;
}

double dilep_m(vector<double> m1, vector<double> m2) {
	return m1[0] + m2[0];
}

double delta_r(double eta1, double eta2, double phi1, double phi2) {
    return std::sqrt( std::pow(eta1 - eta2, 2) + std::pow(phi1 - phi2, 2) );
}

double calc_div(double x1, double x2) {
    return x1 / x2;
}

double calc_diff(double x1, double x2) {
    return std::abs(x1 - x2);
}

#endif
