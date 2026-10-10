from __future__ import annotations
import argparse, hashlib, json, os, random, statistics, tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
DATA_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(exist_ok=True)
DATA_FILE = DATA_DIR / "fashion_mnist_subset.npz"
EXPERIMENT_NAME = "devi-mlops-m6-fashion-mnist"
MODEL_NAME = "fashion-mnist-cnn-classifier"
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_data():
    """Download Fashion-MNIST, then save deterministic subset for DVC."""
    if DATA_FILE.exists():
        print(f"Dataset subset sudah ada: {DATA_FILE}")
        return
    raw = datasets.FashionMNIST(str(DATA_DIR / "raw"), train=True, download=True)
    test_raw = datasets.FashionMNIST(str(DATA_DIR / "raw"), train=False, download=True)
    rng = np.random.default_rng(42)
    # 10,000 train and 2,000 validation from training split; 2,000 test from official test split.
    train_idx = rng.choice(len(raw), size=12000, replace=False)
    test_idx = rng.choice(len(test_raw), size=2000, replace=False)
    x_all = raw.data.numpy()[train_idx]
    y_all = raw.targets.numpy()[train_idx]
    x_test = test_raw.data.numpy()[test_idx]
    y_test = test_raw.targets.numpy()[test_idx]
    # First 10k are training, remaining 2k validation; selected indices are deterministic.
    np.savez_compressed(DATA_FILE, x_train=x_all[:10000], y_train=y_all[:10000],
                        x_val=x_all[10000:], y_val=y_all[10000:],
                        x_test=x_test, y_test=y_test)
    print(f"Subset disimpan: {DATA_FILE}")

def data_hash():
    if not DATA_FILE.exists():
        raise FileNotFoundError("Dataset belum ada. Jalankan: python -m src.main prepare")
    h = hashlib.md5()
    with open(DATA_FILE, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_arrays():
    if not DATA_FILE.exists(): prepare_data()
    z = np.load(DATA_FILE)
    return {k: z[k] for k in z.files}

class FashionCNN(nn.Module):
    def __init__(self, base_channels=16, dropout=0.25, use_bn=True):
        super().__init__()
        c = base_channels
        layers = [nn.Conv2d(1,c,3,padding=1)]
        if use_bn: layers.append(nn.BatchNorm2d(c))
        layers += [nn.ReLU(), nn.MaxPool2d(2),
                   nn.Conv2d(c,c*2,3,padding=1)]
        if use_bn: layers.append(nn.BatchNorm2d(c*2))
        layers += [nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
                   nn.Linear(c*2*7*7, 128), nn.ReLU(), nn.Dropout(dropout),
                   nn.Linear(128, 10)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def make_loader(x, y, batch_size, shuffle=False, augment=False):
    # Augmentation is applied on-the-fly with simple tensor-space flips/affine-free noise.
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(1) / 255.0
    y = torch.tensor(y, dtype=torch.long)
    if augment:
        # Deterministic dataset tensors; light random horizontal shift is avoided for clothing classes.
        x = torch.clamp(x + torch.randn_like(x) * 0.015, 0, 1)
    return DataLoader(list(zip(x, y)), batch_size=batch_size, shuffle=shuffle)

def evaluate(model, loader, device):
    model.eval(); ys=[]; preds=[]; losses=[]
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device)
            logits=model(xb); losses.append(criterion(logits,yb).item())
            preds.extend(logits.argmax(1).cpu().numpy().tolist())
            ys.extend(yb.cpu().numpy().tolist())
    return float(np.mean(losses)), float(accuracy_score(ys,preds)), float(f1_score(ys,preds,average="macro")), ys, preds

def train_model(params, seed=42, epochs=10, log_epochs=True, run=None):
    seed_everything(seed); device=get_device(); a=load_arrays()
    augment=bool(params.get("augment", True))
    train_loader=make_loader(a["x_train"],a["y_train"],int(params.get("batch_size",64)),True,augment)
    val_loader=make_loader(a["x_val"],a["y_val"],int(params.get("batch_size",64)))
    test_loader=make_loader(a["x_test"],a["y_test"],int(params.get("batch_size",64)))
    model=FashionCNN(int(params.get("base_channels",16)),float(params.get("dropout",0.25)),
                     bool(params.get("use_bn",True))).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=float(params.get("lr",1e-3)),
                               weight_decay=float(params.get("weight_decay",1e-4)))
    criterion=nn.CrossEntropyLoss(); best_acc=-1; best_state=None; best_epoch=0
    for epoch in range(1,epochs+1):
        model.train(); train_losses=[]
        for xb,yb in train_loader:
            xb,yb=xb.to(device),yb.to(device); optimizer.zero_grad()
            loss=criterion(model(xb),yb); loss.backward(); optimizer.step()
            train_losses.append(loss.item())
        val_loss,val_acc,val_f1,_,_=evaluate(model,val_loader,device)
        train_loss=float(np.mean(train_losses))
        if run is not None and log_epochs:
            mlflow.log_metric("train_loss",train_loss,step=epoch)
            mlflow.log_metric("val_loss",val_loss,step=epoch)
            mlflow.log_metric("val_acc",val_acc,step=epoch)
            mlflow.log_metric("val_f1_macro",val_f1,step=epoch)
        if val_acc > best_acc:
            best_acc=val_acc; best_epoch=epoch
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    val_loss,val_acc,val_f1,_,_=evaluate(model,val_loader,device)
    test_loss,test_acc,test_f1,y_true,y_pred=evaluate(model,test_loader,device)
    return model, {"train_loss":train_loss,"val_loss":val_loss,"val_acc":val_acc,
                   "val_f1_macro":val_f1,"best_val_acc":best_acc,"best_epoch":best_epoch,
                   "test_loss":test_loss,"test_acc":test_acc,"test_f1_macro":test_f1,
                   "test_correct":int(sum(np.array(y_true)==np.array(y_pred))),
                   "test_total":len(y_true)}, device

def setup_mlflow():
    uri=os.getenv("MLFLOW_TRACKING_URI","http://127.0.0.1:5000")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return uri

def log_context():
    md5=data_hash()
    mlflow.set_tag("data_md5",md5)
    mlflow.log_param("data_md5",md5)
    mlflow.set_tag("student","Devi Rahayu")
    mlflow.set_tag("nim","123450010")
    mlflow.log_param("device",str(get_device()))
    mlflow.log_param("torch_version",torch.__version__)
    mlflow.log_param("data_file",str(DATA_FILE.relative_to(ROOT)))
    return md5

def baseline():
    setup_mlflow()
    params={"lr":1e-3,"batch_size":64,"dropout":0.25,"base_channels":16,
            "weight_decay":1e-4,"augment":True,"use_bn":True,"seed":42}
    with mlflow.start_run(run_name="00-baseline-cnn") as run:
        mlflow.log_params({**params,"epochs":10})
        log_context()
        model,metrics,device=train_model(params,seed=42,epochs=10,run=run)
        mlflow.log_metrics(metrics)
        mlflow.pytorch.log_model(model,"model")
        mlflow.set_tag("stage","baseline")
        print(json.dumps(metrics,indent=2))
        print("run_id:",run.info.run_id)

def tune(n_trials=24):
    import optuna
    setup_mlflow()
    with mlflow.start_run(run_name="optuna-sweep") as parent:
        log_context()
        def objective(trial):
            params={"lr":trial.suggest_float("lr",1e-4,3e-3,log=True),
                    "batch_size":trial.suggest_categorical("batch_size",[32,64,128]),
                    "dropout":trial.suggest_float("dropout",0.1,0.5),
                    "base_channels":trial.suggest_categorical("base_channels",[8,16,24]),
                    "weight_decay":trial.suggest_float("weight_decay",1e-6,1e-3,log=True),
                    "augment":trial.suggest_categorical("augment",[True,False]),
                    "use_bn":trial.suggest_categorical("use_bn",[True,False]),"seed":42}
            with mlflow.start_run(run_name=f"trial-{trial.number:02d}",nested=True):
                mlflow.log_params({**params,"epochs":6,"trial_number":trial.number})
                log_context()
                _,m,_=train_model(params,seed=42,epochs=6,run=mlflow.active_run())
                mlflow.log_metrics(m); mlflow.set_tag("stage","tuning")
                return m["best_val_acc"]
        study=optuna.create_study(direction="maximize",study_name="fashion-mnist-cnn")
        study.optimize(objective,n_trials=n_trials)
        mlflow.log_metric("best_trial_val_acc",float(study.best_value))
        mlflow.log_params({f"best_{k}":v for k,v in study.best_params.items()})
        print("Best score:",study.best_value)
        print("Best params:",study.best_params)
        (ARTIFACT_DIR/"best_params.json").write_text(json.dumps(study.best_params,indent=2))
        mlflow.log_artifact(str(ARTIFACT_DIR/"best_params.json"))

def read_best_params():
    p=ARTIFACT_DIR/"best_params.json"
    if not p.exists():
        raise FileNotFoundError("Jalankan tune terlebih dahulu untuk membuat artifacts/best_params.json")
    params=json.loads(p.read_text()); params["seed"]=42
    return params

def seeds():
    setup_mlflow(); params=read_best_params()
    with mlflow.start_run(run_name="best-config-3seeds") as parent:
        log_context(); mlflow.log_params({**params,"epochs":10,"seeds":"42,123,2026"})
        results=[]
        for seed in [42,123,2026]:
            with mlflow.start_run(run_name=f"seed-{seed}",nested=True):
                mlflow.log_params({**params,"seed":seed,"epochs":10}); log_context()
                model,m,_=train_model(params,seed=seed,epochs=10,run=mlflow.active_run())
                mlflow.log_metrics(m); results.append(m)
        for key in ["best_val_acc","test_acc","test_f1_macro"]:
            vals=[float(r[key]) for r in results]
            mlflow.log_metric(f"{key}_mean",statistics.mean(vals))
            mlflow.log_metric(f"{key}_std",statistics.stdev(vals))
        print("Three-seed results:",json.dumps(results,indent=2))

def ablation():
    setup_mlflow(); base=read_best_params()
    variants={"full":{},"no_augmentation":{"augment":False},
              "no_dropout":{"dropout":0.0},"no_batch_norm":{"use_bn":False}}
    all_rows=[]
    with mlflow.start_run(run_name="ablation-study") as parent:
        log_context()
        for name,change in variants.items():
            vals=[]
            for seed in [42,123,2026]:
                params={**base,**change}
                with mlflow.start_run(run_name=f"{name}-seed-{seed}",nested=True):
                    mlflow.log_params({**params,"seed":seed,"epochs":10,"ablation":name})
                    log_context()
                    _,m,_=train_model(params,seed=seed,epochs=10,run=mlflow.active_run())
                    mlflow.log_metrics(m); vals.append(m["best_val_acc"])
            all_rows.append({"variant":name,"mean_val_acc":statistics.mean(vals),
                             "std_val_acc":statistics.stdev(vals)})
        out=ARTIFACT_DIR/"ablation_results.json"
        out.write_text(json.dumps(all_rows,indent=2))
        mlflow.log_artifact(str(out))
        print(json.dumps(all_rows,indent=2))

def register():
    setup_mlflow(); params=read_best_params()
    client=MlflowClient()
    with mlflow.start_run(run_name="final-model-register") as run:
        log_context(); mlflow.log_params({**params,"epochs":15,"seed":42})
        model,m,device=train_model(params,seed=42,epochs=15,run=mlflow.active_run())
        mlflow.log_metrics(m)
        model_info=mlflow.pytorch.log_model(model,"model",registered_model_name=MODEL_NAME)
        mlflow.set_tag("stage","registered-model")
        print("Metrics:",json.dumps(m,indent=2))
        print("Registered model:",MODEL_NAME)
    # Assign alias to the latest version created by this run.
    versions=client.search_model_versions(f"name='{MODEL_NAME}'")
    matching=[v for v in versions if v.run_id==run.info.run_id]
    if not matching: raise RuntimeError("Model terdaftar tetapi versi tidak ditemukan.")
    version=max(matching,key=lambda v:int(v.version))
    client.set_registered_model_alias(MODEL_NAME,"champion",version.version)
    print(f"Alias champion -> version {version.version}")

def predict():
    setup_mlflow(); a=load_arrays(); device=get_device()
    model=mlflow.pytorch.load_model(f"models:/{MODEL_NAME}@champion", map_location=device)
    loader=make_loader(a["x_test"][:16],a["y_test"][:16],16)
    model.eval()
    with torch.no_grad():
        xb,yb=next(iter(loader)); pred=model(xb.to(device)).argmax(1).cpu().numpy()
    correct=int((pred==yb.numpy()).sum())
    print(f"Prediksi benar: {correct}/16")
    print("Actual:",yb.numpy().tolist())
    print("Predicted:",pred.tolist())

def main():
    p=argparse.ArgumentParser()
    p.add_argument("command",choices=["prepare","baseline","tune","seeds","ablation","register","predict"])
    p.add_argument("--trials",type=int,default=24)
    args=p.parse_args()
    if args.command=="prepare": prepare_data()
    elif args.command=="baseline": baseline()
    elif args.command=="tune": tune(args.trials)
    elif args.command=="seeds": seeds()
    elif args.command=="ablation": ablation()
    elif args.command=="register": register()
    elif args.command=="predict": predict()

if __name__=="__main__": main()
