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

    fig, ax=plt.subplots(3,1)

    #print("train_loss:\n",progress_dict["train_loss"])
    print("MIN train_loss (loss, term_1, term_2, acc):\n",
            min(progress_dict["train_loss"][start_at:]),
            min(progress_dict["train_term_1"][start_at:]),
            min(progress_dict["train_term_2"][start_at:]),
            min(progress_dict["train_acc_hist"][start_at:]))
    #print("val_loss:\n",progress_dict["val_loss"])
    print("MIN val_loss (loss, term_1, term_2):\n",
            min(progress_dict["val_loss"][start_at:]),
            min(progress_dict["val_term_1"][start_at:]),
            min(progress_dict["val_term_2"][start_at:]),
            min(progress_dict["val_acc_hist"][start_at:]))

    ax[0].plot(progress_dict["train_loss"][start_at:],"r",label="train")
    ax[1].plot(progress_dict["train_term_1"][start_at:],"y",label="train term_1 (cluster id)")
    ax[2].plot(progress_dict["train_term_2"][start_at:],"y--",label="train term_2 (target action)")
    ax[2].plot(progress_dict["train_acc_hist"][start_at:],"r",label="train accuracy")
    ax[0].plot(progress_dict["val_loss"][start_at:],"b",label="validation")
    ax[1].plot(progress_dict["val_term_1"][start_at:],"g",label="validation term 1 (cluster id)")
    ax[2].plot(progress_dict["val_term_2"][start_at:],"g--",label="validation term 2 (target action)")
    ax[2].plot(progress_dict["val_acc_hist"][start_at:],"b",label="val accuracy")
    ax[1].set_ylabel("Focal term")
    ax[2].set_ylabel("MSE term")
    lr=progress_dict["lr"] if "lr" in progress_dict.keys() else "not logged"
  
    factor=1.2
    ax[0].set_ylim([0, 1.2*max(max(progress_dict["train_loss"][start_at:]),max(progress_dict["val_loss"][start_at:]))])
    ax[1].set_ylim([0, 1.2*max(max(progress_dict["train_term_1"][start_at:]),max(progress_dict["val_term_1"][start_at:]))])
    ax[2].set_ylim([0, 1.2*max(max(progress_dict["train_term_2"][start_at:]),
        max(progress_dict["val_term_2"][start_at:]),
        max(progress_dict["train_acc_hist"][start_at:]),
        max(progress_dict["val_acc_hist"][start_at:]),
        )])
    
    if len(sys.argv)>3:
        for iii in range(3):
            ax[iii].set_xlim([0,int(sys.argv[3])])
        ax[1].hlines(0.02,0,int(sys.argv[3]),"m",linestyle="dashdot")
        ax[2].hlines(0.005,0,int(sys.argv[3]),"m",linestyle="dashdot")

        #plt.hlines(min(progress_dict["train_loss"][start_at:]),0,int(sys.argv[3]),"r",linestyle="--")
        #plt.hlines(min(progress_dict["val_loss"][start_at:]),0,int(sys.argv[3]),"b",linestyle="--")
    from matplotlib.ticker import MultipleLocator
    ax[1].yaxis.set_major_locator(MultipleLocator(0.1))
    ax[1].tick_params(axis='both', which='major', labelsize=5)
    ax[2].yaxis.set_major_locator(MultipleLocator(0.01))
    ax[2].tick_params(axis='both', which='major', labelsize=5)


    #plt.tight_layout()
    for iii in range(3):
        ax[iii].legend(loc='upper right',fontsize=13)
        ax[iii].grid("on")
    fig.suptitle(f"epoch with best val loss={progress_dict['epoch_with_best_val_loss']}, LR at cur epoch ={np.round(lr,6)})")
    plt.show()
