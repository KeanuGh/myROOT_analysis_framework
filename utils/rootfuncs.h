float calcWeights(float mc_weight, float lumi_sample, float sumw) {
	return (mc_weight * lumi_sample) / sumw;
}

float calcTruthWeights(float weight, float KFactor, float pileup) {
	return weight * KFactor * pileup;
}

bool filterPeak(float inv_mass) {
	return inv_mass < 120.;
}


float getVecVal(ROOT::VecOps::RVec<float> x, int i = 0);

float getVecVal(ROOT::VecOps::RVec<float> x, int i) {
    if (x.size() > i)  return x[i];
    else               return NAN;
}