save_to== /tmp/logdir//behavior_0
py dataset_tools.py --input_archive /tmp/logdir/NS_log_au8ndcGSC7_273123/archive_2  --plot_behavior_space_traj
py  dataset_tools.py --input_archive /tmp/logdir/annotated_archive.pickle  --export_annotation --plot_behavior_space_traj
py -m scoop -n 22 dataset_tools.py  --out_dir /tmp/logdir/ --input_archive /tmp/archive_990 --verify_repeatability
py -i dataset_tools.py --input_archive ~/Desktop/tmp_desktop/train_data_1/annotated_archive.pickle --add_llm_descriptions ~/Desktop/tmp_desktop/description_out/
