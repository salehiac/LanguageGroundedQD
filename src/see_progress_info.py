import matplotlib.pyplot as plt
import sys
import json



if __name__=="__main__":

    fn=sys.argv[1]
    with open(fn,"r") as fl:
        progress_dict=json.load(fl)

    start_at=0
    if len(sys.argv)>2:
        start_at=int(sys.argv[2])

    print("train_loss:\n",progress_dict["train_loss"])
    print("val_loss:\n",progress_dict["val_loss"])
    plt.plot(progress_dict["train_loss"][start_at:],"r",label="train")
    plt.plot(progress_dict["val_loss"][start_at:],"b",label="validation")
    plt.title(f"epoch with best val loss={progress_dict['epoch_with_best_val_loss']}")
    plt.legend(fontsize=16)
    plt.show()
