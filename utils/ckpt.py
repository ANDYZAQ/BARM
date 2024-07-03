import torch
import utils

def save_ckpt(path, model, optimizer, best_score):
    torch.save({
        "model_state": model.module.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_score": best_score,
    }, path)
    print("Model saved as %s" % path)