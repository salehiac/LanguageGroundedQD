import matplotlib.pyplot as plt
import sys
import json



if __name__=="__main__":

    fn=sys.argv[1]
    with open(fn,"r") as fl:
        progress_dict=json.load(fl)


    plt.plot(progress_dict["train_loss"],"r")
    plt.plot(progress_dict["val_loss"],"b")
    plt.title(f"epoch with best val loss={progress_dict['epoch_with_best_val_loss']}")
    plt.show()
