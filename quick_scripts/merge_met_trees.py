from array import array

import ROOT
from numpy import pi

from utils.file_utils import multi_glob


def copy_met(filepath: str) -> None:
    with ROOT.TFile(filepath, "update") as tfile:
        nominal_tree_name = "T_s1thv_NOMINAL"
        nominal_tree = tfile.Get(nominal_tree_name)
        systematic_trees = [
            tree.GetName()
            for tree in tfile.GetListOfKeys()
            if (tree.GetName() != nominal_tree_name) and (tree.GetClassName() == "TTree")
        ]

        # create dictionary of non-nan met values
        max_entries = nominal_tree.GetEntries()
        met_map = dict()
        taupt_map = dict()
        met_phi_map = dict()
        print(f"Running over {max_entries} entries for nominal in file {filepath}...")
        for i in range(max_entries):
            nominal_tree.GetEntry(i)
            if nominal_tree.MET_met != ROOT.nullptr:
                met_map[nominal_tree.eventNumber] = nominal_tree.MET_met
            if nominal_tree.TauPt != ROOT.nullptr:
                taupt_map[nominal_tree.eventNumber] = nominal_tree.TauPt
            if nominal_tree.TauPt != ROOT.nullptr:
                taupt_map[nominal_tree.eventNumber] = nominal_tree.TauPt
            if nominal_tree.MET_phi != ROOT.nullptr:
                met_phi_map[nominal_tree.eventNumber] = nominal_tree.MET_phi

            if i % 1000 == 0:
                print(f"{i} / {max_entries}", end="\r")

        # print(f"met_map has {len(met_map)} entries")

        # fill systematic trees
        print(f"Filling MET systematics in file {filepath}...")
        for sys_name in systematic_trees:
            # print(f"Found systematic tree: {sys_name}")
            sys_tree = tfile.Get(sys_name)

            # create new branch
            met_met_arr = array("f", [0])
            met_phi_arr = array("f", [0])
            new_met_met_branch = sys_tree.Branch("MET_met", met_met_arr, "MET_met/F")
            new_met_phi_branch = sys_tree.Branch("MET_phi", met_phi_arr, "MET_phi/F")

            # fill matching tree
            # nominal tree contains truth data and will always be larger
            max_entries = sys_tree.GetEntries()
            # print(f"Running over {max_entries} entries for systematic tree {sys_name}...")
            for i in range(max_entries):
                if i % 1000 == 0:
                    print(f"{i} / {max_entries}", end="\r")

                sys_tree.GetEntry(i)

                # to account for sys, sys_met = nominal_met - (sys_pt - nominal_pt)
                sys_taupt = sys_tree.TauPt
                nominal_met = met_map.get(sys_tree.eventNumber, ROOT.nullptr)
                nominal_taupt = taupt_map.get(sys_tree.eventNumber, sys_taupt)
                if nominal_met == ROOT.nullptr:
                    met_met_arr[0] = nominal_taupt
                else:
                    sys_diff = sys_taupt - nominal_taupt
                    met_met_arr[0] = max(nominal_met - sys_diff, 0)  # don't let it get negative
                new_met_met_branch.Fill()

                # met_phi is blank, use inverse of tau phi
                met_phi_arr[0] = met_map.get(sys_tree.eventNumber, 2 * pi - sys_tree.TauPhi)
                new_met_phi_branch.Fill()

            # attempt to save the branch into the systematic tree
            sys_tree.Write("", ROOT.TObject.kOverwrite)
            # print(f"Written to tree {sys_tree.GetName()}")


if __name__ == "__main__":
    # file = "test_data/user.kghorban.40997791._000001.histograms.root"
    # copy_met_linear(file)

    files_path = "/data/DTA_outputs/2024-08-28/**/*.root"
    files = multi_glob(files_path)
    for file in files:
        # print(f"\n{file}")
        copy_met(file)
