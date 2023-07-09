import matplotlib.pyplot as plt
import sys
import json
import MiscUtils
import numpy as np



if __name__=="__main__":

    fn=sys.argv[1]
    with open(fn,"r") as fl:
        progress_dict=json.load(fl)

    start_at=0
    if len(sys.argv)>2:
        start_at=int(sys.argv[2])

    #print("train_loss:\n",progress_dict["train_loss"])
    print("MIN train_loss (loss, term_1, term_2):\n",min(progress_dict["train_loss"][start_at:]), min(progress_dict["train_term_1"][start_at:]), min(progress_dict["train_term_2"][start_at:]))
    #print("val_loss:\n",progress_dict["val_loss"])
    print("MIN val_loss (loss, term_1, term_2):\n",min(progress_dict["val_loss"][start_at:]), min(progress_dict["val_term_1"][start_at:]), min(progress_dict["val_term_2"][start_at:]))
    plt.plot(progress_dict["train_loss"][start_at:],"r",label="train")
    plt.plot(progress_dict["train_term_1"][start_at:],"y",label="train term_1 (cluster id)")
    plt.plot(progress_dict["train_term_2"][start_at:],"y--",label="train term_2 (target action)")
    plt.plot(progress_dict["val_loss"][start_at:],"b",label="validation")
    plt.plot(progress_dict["val_term_1"][start_at:],"g",label="validation term 1 (cluster id)")
    plt.plot(progress_dict["val_term_2"][start_at:],"g--",label="validation term 2 (target action)")
    lr=progress_dict["lr"] if "lr" in progress_dict.keys() else "not logged"
    plt.title(f"epoch with best val loss={progress_dict['epoch_with_best_val_loss']}, LR at cur epoch ={np.round(lr,6)})")
    #plt.title(f"min train_loss={min(progress_dict['train_loss'])}\n min val_loss={min(progress_dict['val_loss'])}")
    plt.ylim([0, max(max(progress_dict["train_loss"][start_at:]),max(progress_dict["val_loss"][start_at:]))])
    if len(sys.argv)>3:
        plt.xlim([0,int(sys.argv[3])])
        plt.hlines(0.01,0,int(sys.argv[3]),"m",linestyle="dashdot")

        #plt.hlines(min(progress_dict["train_loss"][start_at:]),0,int(sys.argv[3]),"r",linestyle="--")
        #plt.hlines(min(progress_dict["val_loss"][start_at:]),0,int(sys.argv[3]),"b",linestyle="--")

    plt.tight_layout()
    plt.legend(loc='upper right',fontsize=13)
    plt.grid("on")
    plt.show()
