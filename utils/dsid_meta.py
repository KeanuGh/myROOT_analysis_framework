import json
import logging
import math
import os
import re
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Generator

import ROOT
import pandas as pd

from src.logger import get_logger
from utils.file_utils import multi_glob


@dataclass(slots=True)
class DatasetIdMetaContainer:
    """Container class for dataset metadata"""

    dsid: int = 0
    sumw: float = 1.0
    cross_section: float = 1.0
    kfactor: float = 1.0
    filter_eff: float = 1.0
    phys_short: str = ""
    generator_name: str = ""
    total_events: int = 0
    total_size: str = ""
    etag: str = ""
    ptag: str = ""
    stag: str = ""
    rtag: str = ""

    def __getitem__(self, item):
        return self.__getattribute__(item)


@dataclass(slots=True)
class DatasetMetadata:
    """
    Container class for metadata per dataset ID. Wrapper around a dictionary containing ID: DatasetIdMetaContainer
    key-value pairs.
    """

    logger: logging.Logger = field(default_factory=get_logger)
    dataset_dsids: dict[str, list[int]] = field(init=False, default_factory=dict)
    _dsid_meta_dict: dict[int, DatasetIdMetaContainer] = field(init=False, default_factory=dict)
    _pmg_db: pd.DataFrame = field(init=False)
    _PMG_DB: Path = Path(__file__).parent / "PMGxsecDB_mc16.txt"
    _PMG_DB_BACKUP: Path = Path(
        "/cvmfs/atlas.cern.ch/repo/sw/database/GroupData/dev/PMGTools/PMGxsecDB_mc16.txt"
    )

    def __getitem__(self, dsid: int) -> DatasetIdMetaContainer:
        self.__err_on_no_dict()
        return self._dsid_meta_dict[dsid]

    def __iter__(self) -> Generator[tuple[int, DatasetIdMetaContainer], None, None]:
        self.__err_on_no_dict()
        yield from self._dsid_meta_dict.items()

    def __len__(self) -> int:
        return len(self._dsid_meta_dict)

    def __check_id_dict(self) -> bool:
        return len(self._dsid_meta_dict) > 0

    def __err_on_no_dict(self) -> None:
        if not self.__check_id_dict():
            raise ValueError("Dictionary of DSID metadata is empty. Run `fetch_metadata()` first.")

    def _get_pmg_db(self) -> pd.DataFrame:
        """Return pandas DataFrame containing data drom PMG database"""

        def __read_pmg(file: Path) -> pd.DataFrame:
            """Return DataFrame from database file"""
            return pd.read_csv(
                file,
                delim_whitespace=True,
                header=0,
                index_col=0,
                names=[
                    "DSID",
                    "physics_short",
                    "crossSection",
                    "genFiltEff",
                    "kFactor",
                    "relUncertUP",
                    "relUncertDOWN",
                    "generator_name",
                    "etag",
                ],
                dtype={
                    "DSID": int,
                    "physics_short": str,
                    "crossSection": float,
                    "genFiltEff": float,
                    "kFactor": float,
                    "relUncertUP": float,
                    "relUnvertDOWN": float,
                    "generator_name": str,
                    "etag": str,
                },
            )

        try:
            self.logger.info(f"reading pmg database file '%s'...", self._PMG_DB)
            return __read_pmg(self._PMG_DB)
        except FileNotFoundError:
            try:
                self.logger.info(
                    f"File not found. Looking for backup at '%s'...", self._PMG_DB_BACKUP
                )
                return __read_pmg(self._PMG_DB_BACKUP)
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Could not find PMG database file '{self._PMG_DB}'") from e

    def fetch_metadata(
        self, datasets: dict, ttree: str = "T_s1thv_NOMINAL", data_year: int = 2017
    ) -> None:
        """fetch metadata for current datasets based on dataset IDs in passed ROOT files"""

        if self.__check_id_dict():
            self.logger.info("Regenerating DSID metadata for data-year %s...", data_year)

        # load and test modules first
        try:
            import pyAMI.client as pyami_client
            import pyAMI_atlas.api as atlas_api
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Install pyami_atlas to regenerate metadata cache:\n"
                "run `pip install pyAMI_core pyAMI_atlas`\n"
                "You may need to be running on lxplus."
            ) from e

        # check for voms cert
        try:
            res = subprocess.run(["voms-proxy-info", "-fqan", "-exists"], capture_output=True)
        except FileNotFoundError:
            raise OSError("No voms module found. Are you running on lxplus?")

        if res.returncode:
            self.logger.info("Connecting to voms proxy server..")
            os.system("voms-proxy-init -voms atlas")

        # initialise pyami
        client = pyami_client.Client("atlas-v2")
        atlas_api.init()

        # initialise PMG tool
        pmg_df = self._get_pmg_db()

        # collect files and calculate sumw
        all_files: set[str] = set()
        data_samples: set[int] = set()
        for dataset_name, dataset_dict in datasets.items():
            files = multi_glob(dataset_dict["data_path"])
            ttree_name = dataset_dict["ttree"] if "tree" in dataset_dict else ttree
            all_files |= set(files)
            dsids: set[int] = set()
            merge_dataset = ""

            self.logger.info("Calculating sum of weights for dataset '%s'...", dataset_name)

            # loop over files and sum sumw values per dataset ID (assuming each file only has one dataset ID value)
            for file in files:
                if "is_data" in dataset_dict and dataset_dict["is_data"]:
                    if "period" not in str(file):
                        raise ValueError(f"Are you sure data file {file} is a data sample?")

                    # save data-period as unique integer as pretend dataset ID
                    period = re.compile(r"data[0-9]+\.period(\w)\.").search(file).group(1)
                    dsid = self._period_to_int(data_year, period)

                    dsids.add(dsid)
                    data_samples.add(dsid)
                    if dsid not in self._dsid_meta_dict:
                        self._dsid_meta_dict[dsid] = DatasetIdMetaContainer(dsid=dsid, sumw=0)

                    continue

                with ROOT.TFile(file, "read") as tfile:
                    if not tfile.GetListOfKeys().Contains(ttree_name):
                        raise ValueError(
                            "Missing key '{}' from file {}\nKeys available: {}".format(
                                ttree_name,
                                tfile,
                                "\n".join([key.GetName() for key in tfile.GetListOfKeys()]),
                            )
                        )

                    # read first DSID from branch (there should only be one value)
                    tree = tfile.Get(ttree_name)
                    tree.GetEntry(0)
                    dsid = tree.mcChannel
                    dsids.add(dsid)

                    # handle dataset merging
                    if "merge_into" in dataset_dict:
                        merge_dataset = dataset_dict["merge_into"]
                        if merge_dataset in self.dataset_dsids:
                            if dsid not in self.dataset_dsids[merge_dataset]:
                                self.dataset_dsids[merge_dataset].append(dsid)
                        else:
                            self.dataset_dsids[merge_dataset] = [dsid]

                    sumw = tfile.Get("sumOfWeights").GetBinContent(4)  # bin 4 is AOD sum of weights

                    if dsid == 0:
                        self.logger.warning("Passed a '0' DSID for dataset %s", dataset_name)
                        # workaround for broken wmunu samples and ignore data
                        if "Sh_2211_Wmunu_mW_120_ECMS_BFilter" in str(file):
                            self.logger.warning(
                                f"Working around broken DSID for file {file}, setting DSID to 700446"
                            )
                            dsid = 700446
                        elif "is_data" in dataset_dict and dataset_dict["is_data"]:
                            pass
                        else:
                            self.logger.error(
                                f"Unknown DSID for file {file}, THIS WILL LEAD TO A BROKEN DATASET!!!"
                            )

                    # self.logger.debug(f"dsid: {dsid}: sumw {sumw} for file {file}")
                    if dsid not in self._dsid_meta_dict:
                        self._dsid_meta_dict[dsid] = DatasetIdMetaContainer(dsid=dsid, sumw=sumw)
                    else:
                        self._dsid_meta_dict[dsid].sumw += sumw

            self.dataset_dsids[dataset_name] = list(dsids)
            if merge_dataset:
                self.dataset_dsids[dataset_name] = list(dsids)
            self.logger.debug(
                "Found the following dataset IDs for dataset '%s': %s", dataset_name, dsids
            )
        
        if len(self._dsid_meta_dict) == 0:
            raise ValueError("No datasets found in passed files. Double check inputs.")

        # get other metadata for dataset IDs
        for dsid in self._dsid_meta_dict:
            self.logger.info("Getting metadata for dataset %s...", dsid)

            if dsid in data_samples:  # for data
                year, period = self._int_to_period(dsid)
                year -= 2000  # actually just need the last two numbers
                ds_name = f"data{year}_13TeV.period{period}.physics_Main.PhysCont.DAOD_PHYS.grp{year}_v01_p5314"

                self.logger.info("Fetching info from AMI for dataset: %s", ds_name)
                ds_info = atlas_api.get_dataset_info(client, dataset=ds_name)[0]

                # save info
                self._dsid_meta_dict[dsid].phys_short = ds_name
                self._dsid_meta_dict[dsid].total_events = ds_info["totalEvents"]
                self._dsid_meta_dict[dsid].total_size = self._convert_size(
                    int(ds_info["totalSize"])
                )

            else:  # for MC
                dsid_pmg_data = pmg_df.loc[dsid]
                self._dsid_meta_dict[dsid].cross_section = dsid_pmg_data["crossSection"]
                self._dsid_meta_dict[dsid].kfactor = dsid_pmg_data["kFactor"]
                self._dsid_meta_dict[dsid].filter_eff = dsid_pmg_data["genFiltEff"]
                self._dsid_meta_dict[dsid].phys_short = dsid_pmg_data["physics_short"]
                self._dsid_meta_dict[dsid].etag = dsid_pmg_data["etag"]
                self._dsid_meta_dict[dsid].generator_name = dsid_pmg_data["generator_name"]

                # work out dataset name from dsid
                # e-tag?
                etag = dsid_pmg_data["etag"]

                # r-tag?
                try:
                    rtag = {
                        2015: "r9364",
                        2016: "r9364",
                        2017: "r10201",
                        2018: "r10724",
                    }[data_year]
                except KeyError as e:
                    raise ValueError(
                        f"Missing dataset year: {data_year}. Pass one of ['2015', '2016', '2017', '2018']"
                    ) from e

                # s-tag?
                if "powheg" in dsid_pmg_data["generator_name"].lower():
                    stag = "a875"
                else:
                    stag = "s3126"

                # p-tag?
                possible_ptags = ["p5823", "p5313", "p5083", "p5001"]

                # look for ptags
                short = dsid_pmg_data["physics_short"]
                pattern = f"mc16_13TeV.{dsid}.{short}.deriv.DAOD_PHYS.{etag}_{stag}_{rtag}_p%"

                res = atlas_api.list_datasets(client, patterns=pattern, ami_status="VALID")
                if not res:
                    raise ValueError(f"No matching datasets: {pattern}")

                available_ptags = [ds["ldn"].split("_")[-1] for ds in res if "ldn" in ds]
                ptag_matches = [p for p in possible_ptags if p in available_ptags]
                if not len(ptag_matches):
                    raise ValueError(
                        f"No matching ptags in: {possible_ptags} for dataset: {pattern}"
                    )

                ptag = ptag_matches[0]  # get first (latest) ptag
                ds_name = f"mc16_13TeV.{dsid}.{short}.deriv.DAOD_PHYS.{etag}_{stag}_{rtag}_{ptag}"

                # look for matching dataset
                self.logger.info("Fetching info from AMI for dataset: %s", ds_name)
                ds_info = atlas_api.get_dataset_info(client, dataset=ds_name)[0]

                # save info
                self._dsid_meta_dict[dsid].rtag = rtag
                self._dsid_meta_dict[dsid].stag = stag
                self._dsid_meta_dict[dsid].ptag = ptag
                self._dsid_meta_dict[dsid].total_events = ds_info["totalEvents"]
                self._dsid_meta_dict[dsid].total_size = self._convert_size(
                    int(ds_info["totalSize"])
                )

    @property
    def periods(self) -> dict[int, list[str]]:
        """Data periods"""
        return {
            2015: ["D", "E", "F", "G", "H", "J"],
            2016: ["A", "B", "C", "D", "E", "F", "G", "I", "K", "L"],
            2017: ["B", "C", "D", "E", "F", "H", "I", "K"],
            2018: ["B", "C", "D", "F", "I", "K", "L", "M", "O", "Q"],
        }

    def _period_to_int(self, year: int, period: str) -> int:
        """Convert a data period to a unique integer for caching"""
        return int(f"{year}00{self.periods[year].index(period)}")

    def _int_to_period(self, i: int) -> tuple[int, str]:
        """Convert unique integer back to year and data period"""
        string = str(i)
        year = int(string[:4])
        period = self.periods[int(year)][int(string[-1])]
        return year, period

    def read_metadata(self, file: str | Path) -> None:
        """Get metadata from file"""

        with open(file) as infile:
            json_dict = json.load(infile)

        self._dsid_meta_dict = {
            int(k): DatasetIdMetaContainer(**v) for k, v in json_dict["metadata"].items()
        }
        self.dataset_dsids = json_dict["dataset_ids"]

    def save_metadata(self, file: str | Path) -> None:
        """Save metadata to file"""

        if not self.__check_id_dict():
            raise ValueError("Dictionary of DSID metadata is empty. Run `fetch_metadata` first.")

        dict_to_json = dict()
        dict_to_json["metadata"] = self._dsid_meta_dict
        dict_to_json["dataset_ids"] = self.dataset_dsids

        class DataclassDictConverterEncoder(json.JSONEncoder):
            """Encoder for dataclasses to json"""

            def default(self, obj):
                """."""
                if isinstance(obj, DatasetIdMetaContainer):
                    return asdict(obj)
                return super().default(obj)

        with open(file, "w") as outfile:
            json.dump(dict_to_json, outfile, indent=4, cls=DataclassDictConverterEncoder)

    @staticmethod
    def _convert_size(size_bytes: int) -> str:
        """Convert number of bytes into appropriate string"""
        if size_bytes == 0:
            return "0B"
        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_name[i]}"
