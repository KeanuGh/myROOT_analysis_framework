#include <TFile.h>
#include <TTree.h>

void add_MET_branch() {
  // Open the file containing the nominal and systematic trees
  TFile* file = new TFile("test_data/user.kghorban.40997791._000001.histograms.root", "UPDATE");

  // Get the nominal tree
  const char* nominal_tree_name = "T_s1thv_NOMINAL";
  TTree* nominal_tree;
  file->GetObject(nominal_tree_name, nominal_tree);

  // Get the list of systematic trees
  std::vector<TTree*> systematic_trees;
  TKey *key;
  while ((key = (TKey*)file->GetListOfKeys())) {
    TClass *cl = gROOT->GetClass(key->GetClassName());
    if (!key->InheritsFrom("TTree") | (key->GetName() == nominal_tree_name) ) continue;

    TTree* sys_tree;
    file->GetObject(key->GetName(), sys_tree);
    systematic_trees.push_back(sys_tree);
  }

  // Create map of event number -> met
  std::map<int, float> met_event_map;
  int eventNumber;
  float met;

  int max_entries = nominal_tree->GetEntries();
  std::cout << "Running over nominal.." << std::endl;
  for(int i=0; i<=max_entries; i++){
    nominal_tree->GetEntry(i);
    nominal_tree->SetBranchAddress("MET_met", &met);
    nominal_tree->SetBranchAddress("eventNumber", &eventNumber);

    if (!std::isnan(met)) met_event_map[eventNumber] = met;
  }

  std::cout << "filling systematics..." << std::endl;
  for (TTree* sys_tree: systematic_trees) {

    float met_buffer;
    TBranch* new_met_branch = sys_tree->Branch("MET_met", &met_buffer, "MET_met/F");

    int n_entries = sys_tree->GetEntries();
    int eventNumber;
    for(int i=0; i<=n_entries; i++) {
      nominal_tree->GetEntry(i);
      nominal_tree->SetBranchAddress("eventNumber", &eventNumber);
      met_buffer = met_event_map[eventNumber];
      new_met_branch->Fill();
    }
    sys_tree->Write("", TObject::kOverwrite);
    std::cout << "Added MET_met branch to systematic tree: " << sys_tree->GetName() << std::endl;
  }

  file->Close();
}

add_MET_branch();
