import functools
import argparse
import pickle
import sys
import pdb
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import copy
from wordcloud import WordCloud
from collections import Counter
from typing import List, Any, Literal
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors

import torch
from torch.utils.data import Dataset, DataLoader
from termcolor import colored
from scoop import futures
from deap import tools as deap_tools

import Agents
import NavigationEnv
import NS
import MiscUtils
from environment import create_env_with_objects

_assets = {
    "env_im": "./environment/ressources/maze_19_2.pbm",
    "xml_path": "./environment/maze_setup_5x5.xml"
}


def generate_paper_dataset(logs_root="/tmp/"):
    """
    """

    problem = NavigationEnv.NavigationEnv(bd_type="generic", assets=_assets)

    nov_estimator = NS.ArchiveBasedNoveltyEstimator(k=15)
    arch = NS.ListArchive(max_size=10000,
                          growth_rate=6,
                          growth_strategy="random",
                          removal_strategy="random")

    selector = functools.partial(MiscUtils.selBest,
                                 k=50,
                                 automatic_threshold=False)

    in_dims = problem.dim_obs
    out_dims = problem.dim_act
    num_pop = 50

    normalise_output_with = ""
    num_hidden = 3
    hidden_dim = 10

    def make_ag(idx):
        return Agents.SmallFC_FW(idx,
                                 in_d=in_dims,
                                 out_d=out_dims,
                                 num_hidden=num_hidden,
                                 hidden_dim=hidden_dim,
                                 output_normalisation=normalise_output_with)

    eta = 5
    low = -1.0
    up = 1.0
    indpb = 0.1

    mutator = functools.partial(deap_tools.mutPolynomialBounded,
                                eta=eta,
                                low=low,
                                up=up,
                                indpb=indpb)

    ns = NS.NoveltySearch(archive=arch,
                          nov_estimator=nov_estimator,
                          mutator=mutator,
                          problem=problem,
                          selector=selector,
                          n_pop=num_pop,
                          n_offspring=50,
                          agent_factory=make_ag,
                          visualise_bds_flag=True,
                          map_type="scoop",
                          logs_root=logs_root,
                          initial_pop=[],
                          problem_sampler=None)

    stop_on_reaching_task = False
    nov_estimator.log_dir = ns.log_dir_path
    ns.save_archive_to_file = True

    _, _ = ns(iters=1000, stop_on_reaching_task=False, save_frequency=10)


def find_duplicates(arch):

    dupes = []
    for ii in range(len(arch)):
        for jj in range(ii + 1, len(arch)):

            if arch[ii] == arch[jj]:
                dupes.append((ii, jj))

    return dupes


def make_navigation_env():

    return NavigationEnv.NavigationEnv(bd_type="generic", assets=_assets)


class ArchDataset(Dataset):
    def __init__(self, arch: List[Agents.SmallFC_FW],
                 split: Literal["train", "val", "test"]):

        self.ds = [x.to_batch_example() for x in arch]
        self.split = split

    def __len__(self):

        return len(self.ds)

    def __getitem__(self, idx):

        return self.ds[idx]

    def make_data_loader(self, batch_size):
        return DataLoader(self,
                          batch_size=batch_size,
                          shuffle=True if self.split == "train" else False)


def cluster_actions(per_dim: bool,
                    arch,
                    num_clusters: list,
                    read_clusters_from_file="",
                    save_centers_to="/tmp/",
                    display=True):

    all_actions = []

    for ag_i in range(len(arch)):
        traj = arch[ag_i]._tau
        all_actions.append(traj["action"])

    all_actions = np.concatenate(all_actions, 0)
    np.random.shuffle(all_actions)

    if not per_dim:
        if not read_clusters_from_file:
            kmeans = KMeans(n_clusters=num_clusters[0],
                            random_state=0).fit(all_actions)
        else:
            print("loading cluster centers from ", read_clusters_from_file)
            with open(read_clusters_from_file, "rb") as fl:
                kmeans = pickle.load(fl)

        print('Cluster centers:\n', kmeans.cluster_centers_)

        if display:
            sampled_actions = all_actions[:50000]
            y_pred = kmeans.predict(sampled_actions)

            plt.scatter(sampled_actions[:, 0], sampled_actions[:, 1], c=y_pred)
            plt.scatter(kmeans.cluster_centers_[:, 0],
                        kmeans.cluster_centers_[:, 1],
                        s=300,
                        c='red')  # cluster centers
            plt.show()

        if save_centers_to and not read_clusters_from_file:
            with open(f"{save_centers_to}/cluster_centers.pkl", "wb") as fl:
                pickle.dump(kmeans, fl)

        return [kmeans]

    else:

        if read_clusters_from_file:
            print("loading cluster centers from ", read_clusters_from_file)
            with open(read_clusters_from_file, "rb") as fl:
                kmeans_lst = pickle.load(fl)

        else:
            action_dims = all_actions.shape[1]
            kmeans_lst = []
            assert len(
                num_clusters
            ) == action_dims, "there should be K_i clusters for action dim a_i"
            for d_i in range(action_dims):
                kmeans_i = KMeans(n_clusters=num_clusters[d_i],
                                  random_state=0).fit(all_actions[:,
                                                                  d_i].reshape(
                                                                      -1, 1))
                kmeans_lst.append(kmeans_i)

        if display:  #only works for 2d
            mm = kmeans_lst[0]
            nn = kmeans_lst[1]
            xxx, yyy = np.meshgrid(mm.cluster_centers_, nn.cluster_centers_)
            xxx = xxx.reshape(1, -1)
            yyy = yyy.reshape(1, -1)

            plt.plot(xxx, yyy, "ro")

            #display sampled pts
            sampled_actions = all_actions[:50000]
            y_pred_mm = mm.predict(
                sampled_actions[:, [0]]
            )  #list of indexes in 0, ..., num_clusters_0-1
            y_pred_nn = nn.predict(
                sampled_actions[:, [1]]
            )  #list of indexes in 0, ..., num_clusters_1-1

            y_pred_2d = list(zip(y_pred_nn, y_pred_mm))

            colormap = mcolors.ListedColormap(np.random.rand(256, 3))
            y_colors = list(
                map(lambda x: num_clusters[1] * x[0] + x[1], y_pred_2d))

            plt.scatter(sampled_actions[:, 0],
                        sampled_actions[:, 1],
                        c=y_colors,
                        cmap=colormap)
            plt.show()

        if save_centers_to and not read_clusters_from_file:
            with open(f"{save_centers_to}/cluster_centers.pkl", "wb") as fl:
                pickle.dump(kmeans_lst, fl)

        return kmeans_lst


class ActionAsCenterPlusOffset:
    def __init__(self, kmeans):
        self.kmeans = kmeans

    def transform(self, actions):
        cluster_ids = self.kmeans.predict(actions)
        centers = self.kmeans.cluster_centers_[cluster_ids]
        offsets = actions - centers

        return cluster_ids, offsets

    def reverse(self, cluster_ids, offsets):

        actions = self.kmeans.cluster_centers_[cluster_ids] + offsets
        return actions


class ActionAsCenterPlusOffsetPerDim:
    def __init__(self, kmeans_lst):
        self.kmeans_lst = kmeans_lst

    def transform(self, actions):

        dims = len(self.kmeans_lst)
        assert dims == actions.shape[1]
        offsets = np.zeros_like(actions)
        cluster_ids = np.zeros_like(actions).astype("long")
        for a_i in range(dims):

            pred_i = self.kmeans_lst[a_i].predict(
                actions[:, [a_i]]
            )  #cluster indexes, should be of size (actions.shape[0],)
            centers_i = self.kmeans_lst[a_i].cluster_centers_[pred_i]

            offsets[:, [a_i]] = actions[:, [a_i]] - centers_i
            cluster_ids[:, a_i] = pred_i

        return cluster_ids, offsets

    def reverse(self, cluster_ids, offsets):

        dims = len(self.kmeans_lst)
        assert dims == offsets.shape[1]
        assert dims == cluster_ids.shape[1]
        actions = np.zeros_like(offsets)
        for a_i in range(dims):
            center_i = self.kmeans_lst[a_i].cluster_centers_[cluster_ids[:,
                                                                         a_i]]
            actions[:, [a_i]] = center_i + offsets[:, [a_i]]

        return actions


if __name__ == "__main__":

    _parser = argparse.ArgumentParser(
        description=
        'Tool to generate/annotate datasets for the predefined navigation env.'
    )
    _parser.add_argument(
        '--generate_archive',
        action='store_true',
        help=
        "generate and archive of policies/trajectories/behaviors (results will be saved to --out_dir). NOTE: this should be launched with scoop, e.g. python3 -m scoop -n 32 <this_script>"
    )
    _parser.add_argument('--input_archive',
                         type=str,
                         default="",
                         help="archive to process")
    #those are only relevant if an input archive is given
    _parser.add_argument(
        '--annotate_archive',
        action='store_true',
        help=
        "annotate input archive with semantic info (pre-LLM), results will be saved to a new archive in --out_dir"
    )
    _parser.add_argument(
        '--verify_repeatability',
        action='store_true',
        help=
        "verify that the policies in the input archive produce the {(obs,action)} and behaviors that are in the archive."
    )
    _parser.add_argument(
        '--plot_behavior_space_traj',
        action='store_true',
        help=
        "if given, a figure of each individual's behavior is saved to --out_dir"
    )

    _parser.add_argument(
        '--export_annotations',
        action='store_true',
        help=
        "Will throw an exception if the input archive has not already been annotated. Exports annotation data to --out_dir"
    )
    _parser.add_argument(
        '--add_llm_descriptions',
        type=str,
        default="",
        help=
        "path to a directory with files <filename>_<traj_idx>.json, each containing a dictionary with key 'descr', containing an llm generated description for traj <traj_id>. This argument will add that description to the corresponding policies, and save the archive to --out_dir."
    )

    _parser.add_argument(
        '--visualize_llm_description_and_behavior',
        type=int,
        default=-1,
        help=
        "if its value is a negative value N, then randomly samples and visualizes -N elements. if it's positive, then it is interpreted as an index in the archive, and only that element is visualized. Note however that -1 will be ignored."
    )

    _parser.add_argument('--archive_info',
                         action='store_true',
                         help="prints info about --input_archive")

    _parser.add_argument(
        '--fix_duplicates',
        action='store_true',
        help=
        "Some policies can appear multiple times in the archive. We separate them as a form of data augmentation: this pairs the same policy with different but similar texts"
    )

    _parser.add_argument(
        '--split_archive',
        metavar="x",
        nargs=3,  #the three arguments should be in [0,1] and sum to 1.0
        type=float,
        help=
        "splits archive into train/val/test splits. The three input values should be normalized percentages. For example, values of [0.9, 0.05, 0.05] will keep 90%, 5%, 5% of the data as respectively train, val and test splits. The results are saved to --out_dir"
    )

    _parser.add_argument(
        '--export_as_prompt_list',
        action="store_true",
        help="saves llm description and bds in json format in --out_dir")

    _parser.add_argument(
        '--filter_bds',
        metavar="x",
        nargs=4,  #the four arguments should be min_x, max_x, min_y, max_y
        type=float,
        help=
        "removes behavior descriptors that fall outside at least one of the specified range from the archive"
    )

    _parser.add_argument(
        '--merge_archives',
        metavar="x",
        nargs=2,  #path to two archives to merge 
        type=str,
        help="the two archives are concatenated and then shuffled")

    _parser.add_argument('--shuffle_archive', action="store_true", help="")

    _parser.add_argument(
        '--clamp_actions',
        action="store_true",
        help=
        "if true, limits actions to the range that is tolerated by the gym env"
    )

    _parser.add_argument(
        "--prepare_actions_for_multimodal",
        type=int,
        nargs="*",
        help=
        "If prepare_actions_for_multimodal is not None, then performs k-means with prepare_actions_for_multimodal centroids, and expresses each action as an index and an offset. If only one arg N is passed, kmeans is performed jointly across all action dims. If multiple values are provided, it is assumed that they each correspond to an action dim and a kmeans clustering will be performed for each action dimension."
    )

    _parser.add_argument("--read_kmeans_from_file",
                         type=str,
                         default="",
                         help="")

    _parser.add_argument(
        '--check_dataset_intersections',
        metavar="x",
        nargs=2,  #path to two archives to merge 
        type=str,
        help=
        "sanity check tool to make sure that there are not intersections between the train/val/test splits"
    )

    _parser.add_argument(
        '--check_only_for_same_prompts',
        action="store_true",
        help="",
    )

    #output arg
    _parser.add_argument('--out_dir',
                         type=str,
                         default="/tmp/logdir/",
                         help="directory to save results")

    _args = _parser.parse_args()

    if len(sys.argv) == 1:
        _parser.print_help()

    if _args.generate_archive:
        generate_paper_dataset(_args.out_dir)
    if _args.input_archive:
        with open(_args.input_archive, "rb") as fl:
            _in_arch = pickle.load(fl)

        if _args.plot_behavior_space_traj:

            scene = create_env_with_objects("./environment/")
            list(
                futures.map(
                    lambda x: scene.display(display_bbox=False,
                                            hold_on=False,
                                            path2d_info=(x[0]._behavior, 600,
                                                         600),
                                            save_to=f"{_args.out_dir}/{x[1]}"),
                    zip(_in_arch,
                        [f"behavior_{i}" for i in range(len(_in_arch))])))

        if _args.export_annotations:

            if not hasattr(_in_arch[0], "_annotation"):
                raise Exception(
                    "--export_annotations can only be used with an annotated archive. Did you mean to pass --annotate_archive?"
                )

            for ag_i in range(len(_in_arch)):
                fn = f"{_args.out_dir}/annotation_{ag_i}.json"
                with open(fn, "w") as fl:
                    json.dump(_in_arch[ag_i]._annotation, fl)

        if _args.add_llm_descriptions:

            in_fns = {
                MiscUtils.get_trailing_number(x[:-5]):
                _args.add_llm_descriptions + "/" + x
                for x in os.listdir(_args.add_llm_descriptions)
            }
            for ag_i in range(len(_in_arch)):
                if ag_i in in_fns:
                    print(
                        colored(f"adding description to agent {ag_i}.",
                                "green",
                                attrs=["bold"]))
                    with open(in_fns[ag_i], "r") as fl:
                        descr_string = json.load(fl)["descr"]

                    if "<DESCRIPTOR>" in descr_string:
                        descr_string = descr_string.replace(
                            "<DESCRIPTOR>",
                            f"x={int(_in_arch[ag_i]._behavior_descr.round(0)[0,0])}, y={int(_in_arch[ag_i]._behavior_descr.round(0)[0,1])}"
                        )
                    if "file_cabinet" in descr_string:
                        descr_string = descr_string.replace(
                            "file_cabinet", "file cabinet")
                    _in_arch[ag_i]._llm_descr = descr_string
                else:
                    pass
                    #print(colored(f"no description for agent {ag_i}, skipping it.","red",attrs=["bold"]))

            mm = [
                True if (hasattr(_in_arch[ii], "_llm_descr")
                         and _in_arch[ii]._llm_descr is not None) else False
                for ii in range(len(_in_arch))
            ]
            if False in mm:
                first_non_annotated = mm.index(False)
                assert sum(mm[:first_non_annotated]) == len(
                    mm[:first_non_annotated]
                ), "the llm descr process is incremental w.r.t policy indexes"
                assert sum(
                    mm[first_non_annotated:]
                ) == 0, "the llm descr process is incremental w.r.t policy indexes"

            out_fn = f"{_args.out_dir}/described_archive.pickle"
            print(f"saving to {out_fn}")
            with open(out_fn, "wb") as fl:
                pickle.dump(_in_arch, fl)

        if _args.annotate_archive:
            scene = create_env_with_objects("./environment/")

            _ag_counter = 0
            for ag in _in_arch:

                print(
                    f"annotating trajectory of agent {_ag_counter}/{len(_in_arch)}"
                )
                ag._annotation = scene.annotate_traj(
                    ag._behavior, real_w=600, real_h=600, step=40
                )  #note that we annotate the behavior space trajectory, not tau

                #fig,_=scene.display(display_bbox=False,hold_on=True,path2d_info=(path2d,600,600))
                #plt.show()

                _ag_counter += 1

            _annotated_archive_path = f"{_args.out_dir}/annotated_archive.pickle"
            with open(f"{_annotated_archive_path}", "wb") as fl:
                pickle.dump(_in_arch, fl)
            print(
                colored(
                    f"annotated archive was saved to {_annotated_archive_path}",
                    "green",
                    attrs=["bold"]))

        if _args.visualize_llm_description_and_behavior != -1:

            ### negative inputs except -1 have the special meaning that -1*the_negative_input random samples will be displayed.

            if _args.visualize_llm_description_and_behavior >= 0:
                _idx_lst = [_args.visualize_llm_description_and_behavior]
            else:
                _num_s = -_args.visualize_llm_description_and_behavior
                _idx_lst = np.random.choice(list(range(len(_in_arch))),
                                            _num_s,
                                            replace=False)

            scene = create_env_with_objects("./environment/")

            for _idx in _idx_lst:
                if not hasattr(
                        _in_arch[_idx],
                        "_llm_descr") or _in_arch[_idx]._llm_descr is None:
                    raise Exception(
                        colored(
                            f"Element {_idx} has not been described. See the input options for annotation archives and adding LLM descriptions",
                            "red",
                            attrs=["bold"]))

                fig, _ = scene.display(display_bbox=False,
                                       hold_on=True,
                                       path2d_info=(_in_arch[_idx]._behavior,
                                                    600, 600))

                fig.suptitle(MiscUtils.add_newlines(_in_arch[_idx]._llm_descr))
                plt.tight_layout()
                plt.show()

        if _args.archive_info:

            import pprint
            _sz = len(_in_arch)
            _dd = {}
            _dd["size"] = _sz
            _dd["num annotated"] = functools.reduce(
                lambda acc, x: acc + 1
                if (hasattr(x, "_annotation") and x._annotation is not None)
                else acc, _in_arch, 0)
            _dd["num described"] = functools.reduce(
                lambda acc, x: acc + 1
                if (hasattr(x, "_llm_descr") and x._llm_descr is not None) else
                acc, _in_arch, 0)

            mm = [
                True if (hasattr(x, "_llm_descr") and x._llm_descr is not None)
                else False for x in _in_arch
            ]
            if False in mm:
                first_non_annotated = mm.index(False)
                assert sum(mm[:first_non_annotated]) == len(
                    mm[:first_non_annotated]
                ), "the llm descr process is incremental w.r.t policy indexes"
                assert sum(
                    mm[first_non_annotated:]
                ) == 0, "the llm descr process is incremental w.r.t policy indexes"
            else:
                first_non_annotated = None

            _dd["described from/to"] = f"[0,{first_non_annotated})" if first_non_annotated is not None else "fully described"
            if isinstance(_in_arch[0]._tau["action"], np.ndarray):
                _dd["episode length"] = _in_arch[0]._tau["action"].shape[0]
                _dd["cmd dims"] = _in_arch[0]._tau["action"].shape[1]
                _dd["kmeans representation"] = False
            else:  #cluser center+offset repr
                assert isinstance(_in_arch[0]._tau["action"], tuple)
                _dd["episode length"] = _in_arch[0]._tau["action"][1].shape[0]
                _dd["cmd dims"] = _in_arch[0]._tau["action"][1].shape[1]
                _dd["kmeans representation"] = True

            _dd["obs dims"] = _in_arch[0]._tau["obs"].shape[1]
            _dd["bd dims"] = _in_arch[0]._behavior_descr.shape[1]

            _thresh = 3
            if not False in mm:
                from tokenizers import pre_tokenizers
                pre_tok = pre_tokenizers.Whitespace()

                word_lst = []
                for _ag in _in_arch:
                    word_lst += [
                        x[0] for x in pre_tok.pre_tokenize_str(
                            _ag._llm_descr.lower())
                    ]
                word_lst_unique = list(set(word_lst))

                _dd["num unique words (pretokens)"] = len(word_lst_unique)

                cnt = Counter(word_lst)
                zz = sorted([(k, v) for k, v in cnt.items()],
                            key=lambda x: x[1],
                            reverse=True)
                _dd[f"num pretokens appearing more than {_thresh} times"] = len(
                    [x for x in zz if x[1] > _thresh])

                import warnings
                warnings.filterwarnings("ignore")
                word_dict = dict(zz)
                wordcloud = WordCloud(
                    width=1000,
                    height=500).generate_from_frequencies(word_dict)
                wc_path = '/tmp/wordcloud.png'
                wordcloud.to_file(wc_path)
                print(
                    colored(f"word cloud saved to {wc_path}",
                            "cyan",
                            attrs=["bold"]))
            else:
                _dd["num unique words (pretokens)"] = "N.A"
                _dd[f"num pretokens appearing more than {_thresh} times"] = "N.A"

            _pp = pprint.PrettyPrinter(indent=4, sort_dicts=False)
            _pp.pprint(_dd)

        if _args.fix_duplicates:

            #_dupes=find_duplicates(_in_arch)
            #print(_dupes)

            _in_arch_cp = [copy.deepcopy(x) for x in _in_arch]

            with open(f"{_args.out_dir}/archive_fixed_duplicates.pkl",
                      "wb") as fl:
                pickle.dump(_in_arch_cp, fl)

        if _args.split_archive:

            np.random.shuffle(_in_arch)
            _r_train = _args.split_archive[0]
            _r_val = _args.split_archive[1]
            _r_test = _args.split_archive[2]

            assert _r_train + _r_val + _r_test == 1.0

            _num = len(_in_arch)
            _idx_1 = int(_r_train * _num)
            _idx_2 = _idx_1 + int(_r_val * _num) + 1

            _train_arch = _in_arch[:_idx_1]
            _val_arch = _in_arch[_idx_1:_idx_2]
            _test_arch = _in_arch[_idx_2:]

            basename = _args.input_archive.split("/")[-1].split(".")[0]
            out_f_train = f"{_args.out_dir}/{basename}_train.pkl"
            out_f_val = f"{_args.out_dir}/{basename}_val.pkl"
            out_f_test = f"{_args.out_dir}/{basename}_test.pkl"

            def dump_arch(arch, fn, msg=""):
                with open(fn, "wb") as fl:
                    pickle.dump(arch, fl)
                if msg:
                    print(msg)

            dump_arch(_train_arch,
                      out_f_train,
                      msg=colored(f"wrote train archive to {out_f_train}",
                                  "cyan",
                                  attrs=["bold"]))
            dump_arch(_val_arch,
                      out_f_val,
                      msg=colored(f"wrote val archive to {out_f_val}",
                                  "cyan",
                                  attrs=["bold"]))
            dump_arch(_test_arch,
                      out_f_test,
                      msg=colored(f"wrote test archive to {out_f_test}",
                                  "cyan",
                                  attrs=["bold"]))

        if _args.export_as_prompt_list:

            _pl = [[
                xx._llm_descr,
                np.round(xx._behavior_descr[0, 0], 1),
                np.round(xx._behavior_descr[0, 1], 1)
            ] for xx in _in_arch]
            with open(f"{_args.out_dir}/prompt_file.json", "w") as fl:
                json.dump(_pl, fl)

        if _args.filter_bds:

            _min_x = _args.filter_bds[0]
            _max_x = _args.filter_bds[1]

            _min_y = _args.filter_bds[2]
            _max_y = _args.filter_bds[3]

            _arch_filtered = [
                x for x in _in_arch
                if (x._behavior_descr[0, 0] > _min_x and x._behavior_descr[
                    0, 0] < _max_x) or (x._behavior_descr[0, 1] > _min_y
                                        and x._behavior_descr[0, 1] < _max_y)
            ]

            with open(f"{_args.out_dir}/filtered_archive.pkl", "wb") as fl:
                pickle.dump(_arch_filtered, fl)

        if _args.prepare_actions_for_multimodal:

            _kmeans_file = _args.read_kmeans_from_file if _args.read_kmeans_from_file else ""

            if len(_args.prepare_actions_for_multimodal) == 1:
                _kmeans = cluster_actions(
                    per_dim=False,
                    arch=_in_arch,
                    num_clusters=_args.prepare_actions_for_multimodal,
                    read_clusters_from_file=_kmeans_file)[0]

                aacpo = ActionAsCenterPlusOffset(_kmeans)

                for ag_i in range(len(_in_arch)):

                    if ag_i % 50 == 0:
                        print(f"processed {ag_i}/{len(_in_arch)}")

                    actions_idx_and_offset_repr = aacpo.transform(
                        _in_arch[ag_i]._tau["action"])
                    assert (np.abs(
                        aacpo.reverse(*(actions_idx_and_offset_repr)) -
                        _in_arch[ag_i]._tau["action"]) < 1e-7).all()
                    _in_arch[ag_i]._tau["action"] = actions_idx_and_offset_repr

                with open(f"{_args.out_dir}/archive_transformed_actions.pkl",
                          "wb") as fl:
                    pickle.dump(_in_arch, fl)

            else:
                _kmeans_lst = cluster_actions(
                    per_dim=True,
                    arch=_in_arch,
                    num_clusters=_args.prepare_actions_for_multimodal,
                    read_clusters_from_file=_kmeans_file
                )  ###_kmeans_lst will be a list length act_dim, with _kmeans_lst[i] a KMeans object

                aacpo_pd = ActionAsCenterPlusOffsetPerDim(_kmeans_lst)

                for ag_i in range(len(_in_arch)):

                    if ag_i % 50 == 0:
                        print(f"processed {ag_i}/{len(_in_arch)}")

                    idx_and_offset = aacpo_pd.transform(
                        _in_arch[ag_i]._tau["action"])
                    assert (np.abs(
                        aacpo_pd.reverse(*(idx_and_offset)) -
                        _in_arch[ag_i]._tau["action"]) < 1e-7).all()
                    _in_arch[ag_i]._tau["action"] = idx_and_offset

                with open(f"{_args.out_dir}/archive_transformed_actions.pkl",
                          "wb") as fl:
                    pickle.dump(_in_arch, fl)

        if _args.shuffle_archive:

            np.random.shuffle(_in_arch)

            with open(f"{_args.out_dir}/shuffled_archive.pkl", "wb") as fl:
                pickle.dump(_in_arch, fl)
            print("wrote shuffled archive to --out_dir")

        if _args.clamp_actions:

            _env = make_navigation_env()
            _clamp_msg = "the clamp_action option is only implemented for actions where all action dimensions have the same range. Please extend the implementation if needed."
            assert (_env.env.action_space.high == _env.env.action_space.high[0]
                    ).all(), _clamp_msg
            assert (_env.env.action_space.low == _env.env.action_space.low[0]
                    ).all(), _clamp_msg

            for ag in _in_arch:
                ag._tau["action"] = np.clip(ag._tau["action"],
                                            _env.env.action_space.low[0],
                                            _env.env.action_space.high[0])

            with open(f"{_args.out_dir}/archive_clamped.pkl", "wb") as fl:
                pickle.dump(_in_arch, fl)
            print("wrote clamped archive to --out_dir")

    if _args.merge_archives:

        with open(_args.merge_archives[0], "rb") as fl:
            _in_arch_0 = pickle.load(fl)
        with open(_args.merge_archives[1], "rb") as fl:
            _in_arch_1 = pickle.load(fl)

        _arch = _in_arch_0 + _in_arch_1
        np.random.shuffle(_arch)

        _out_n = f"{_args.out_dir}/merged_archive.pkl"
        print(f"saving archive to {_out_n}...")
        with open(_out_n, "wb") as fl:
            pickle.dump(_arch, fl)

    if _args.check_dataset_intersections:

        with open(_args.check_dataset_intersections[0], "rb") as fl:
            _in_arch_0 = pickle.load(fl)
        with open(_args.check_dataset_intersections[1], "rb") as fl:
            _in_arch_1 = pickle.load(fl)

        _intersections = []
        _num_total = len(_in_arch_0) * len(_in_arch_1)
        _num_checked = 0
        for ii in range(len(_in_arch_0)):
            for jj in range(len(_in_arch_1)):

                if _num_checked and _num_checked % 1000:
                    print(f"checked {_num_checked}/{_num_total}")

                ag_i = _in_arch_0[ii]
                ag_j = _in_arch_1[jj]

                a_cond = ag_i._llm_descr == ag_j._llm_descr
                b_cond = np.allclose(ag_i._behavior_descr,
                                     ag_j._behavior_descr)
                c_cond = np.allclose(ag_i._tau["obs"], ag_j._tau["obs"])
                d_cond = np.allclose(ag_i._tau["action"][0],
                                     ag_j._tau["action"][0])
                e_cond = np.allclose(ag_i._tau["action"][1],
                                     ag_j._tau["action"][1])

                if _args.check_only_for_same_prompts:
                    if a_cond:
                        _intersections.append((ii, jj))

                if a_cond and b_cond and c_cond and d_cond:
                    _intersections.append((ii, jj))

                _num_checked += 1

        print(f"there were {len(_intersections)} intersecting pairs")
        with open(f"{_args.out_dir}/duplicates.json", "w") as fl:
            json.dump(_intersections, fl)
