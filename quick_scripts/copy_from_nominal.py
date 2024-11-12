import multiprocessing as mp

import ROOT

from utils.file_utils import multi_glob


def copy_from_nominal(filepath: str, variables: list[str] | str, exist_ok: bool = False) -> None:
    """Copy variables from nominal tree into all systematic trees"""
    with ROOT.TFile(filepath, "update") as tfile:
        nominal_tree_name = "T_s1thv_NOMINAL"
        nominal_tree = tfile.Get(nominal_tree_name)
        systematic_trees = [
            tree.GetName()
            for tree in tfile.GetListOfKeys()
            if (tree.GetName() != nominal_tree_name) and (tree.GetClassName() == "TTree")
        ]
        if isinstance(variables, str):
            variables = [variables]

        # check if branches already exist
        if not exist_ok:
            branches_exist = False
            for sys in systematic_trees:
                sys_tree = tfile.Get(sys)
                branches = sys_tree.GetListOfBranches()
                if all([v in branches for v in variables]):
                    branches_exist = True
            if branches_exist:
                print(f"Branches exist in file: {filepath}. Exiting.")
                return

        # create dictionary of non-nan met values
        max_entries = nominal_tree.GetEntries()
        variables_map: dict[str, dict[int, float]] = {v: dict() for v in variables}
        print(f"Running over {max_entries} entries for nominal in file {filepath}...")
        for i in range(max_entries):
            nominal_tree.GetEntry(i)
            for v in variables:
                try:
                    if (val := nominal_tree.__getattr__(v)) != ROOT.nullptr:
                        variables_map[v][nominal_tree.eventNumber] = val
                except AttributeError as e:
                    raise AttributeError(
                        f"No Branch '{v}' found in tree '{nominal_tree.GetName()}' of file: {filepath}"
                    ) from e

            if i % 1000 == 0:
                print(f"{i} / {max_entries} in file {filepath}", end="\r")

        # fill systematic trees
        print(f"Copying variables: {variables} to systematics in file {filepath}...")
        for sys_name in systematic_trees:
            # print(f"Found systematic tree: {sys_name}")
            sys_tree = tfile.Get(sys_name)

            # create new branch
            arrays: dict[str, ROOT.TArray] = {}
            branches: dict[str, ROOT.TBranch] = {}
            for v in variables:
                arrays[v] = ROOT.TArrayD(1)
                branches[v] = sys_tree.Branch(v, arrays[v], f"{v}/D")

            # fill matching tree
            # nominal tree contains truth data and will always be larger
            max_entries = sys_tree.GetEntries()
            # print(f"Running over {max_entries} entries for systematic tree {sys_name}...")
            for i in range(max_entries):
                if i % 1000 == 0:
                    print(f"{i} / {max_entries}", end="\r")

                sys_tree.GetEntry(i)
                for v in variables:
                    arrays[v][0] = variables_map.get(sys_tree.eventNumber, 0)

            # attempt to save the branch into the systematic tree
            sys_tree.Write("", ROOT.TObject.kOverwrite)
            # print(f"Written to tree {sys_tree.GetName()}")


if __name__ == "__main__":
    # file = "test_data/user.kghorban.40997791._000001.histograms.root"
    # copy_met_linear(file)

    files_path = "/data/DTA_outputs/2024-09-19/**/*.root"
    n_workers = mp.cpu_count() // 2
    files = [f for f in multi_glob(files_path) if "user.kghorban.data" not in f]
    with mp.Pool(n_workers) as pool:
        # print(f"\n{file}")
        pool.starmap(copy_from_nominal, [(file, "TruthBosonM") for file in files])
