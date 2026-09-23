from pathlib import Path
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt, torch, torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
from statsmodels.datasets import co2
torch.manual_seed(42); torch.set_num_threads(1); s=co2.load_pandas().data['co2'].resample('W').mean().interpolate().astype('float32'); vals=s.to_numpy(); mu,sd=float(vals.mean()),float(vals.std()); z=(vals-mu)/sd; L=24; X=np.array([z[i:i+L] for i in range(len(z)-L)],dtype='float32')[:,:,None]; y=np.array([z[i+L] for i in range(len(z)-L)],dtype='float32')[:,None]; c=int(.8*len(X)); dl=DataLoader(TensorDataset(torch.tensor(X[:c]),torch.tensor(y[:c])),64,shuffle=False)
class M(nn.Module):
 def __init__(self): super().__init__(); self.r=nn.LSTM(1,32,batch_first=True); self.f=nn.Linear(32,1)
 def forward(self,x): o,_=self.r(x); return self.f(o[:,-1])
m=M(); opt=torch.optim.Adam(m.parameters(),lr=.005); lf=nn.MSELoss(); losses=[]
for _ in range(20):
 ls=[]
 for xb,yb in dl: opt.zero_grad(); loss=lf(m(xb),yb); loss.backward(); opt.step(); ls.append(loss.item())
 losses.append(float(np.mean(ls)))
with torch.no_grad(): p=m(torch.tensor(X[c:])).numpy().ravel()*sd+mu; truth=y[c:].ravel()*sd+mu
rmse=float(np.sqrt(np.mean((p-truth)**2))); mae=float(np.mean(np.abs(p-truth))); out={'rmse':rmse,'mae':mae,'window_weeks':L,'n_windows':int(len(X))}; Path('results').mkdir(exist_ok=True); Path('results/metrics.json').write_text(json.dumps(out,indent=2))
plt.figure(figsize=(9,5)); plt.plot(s.index,s.values); plt.xlabel('Date'); plt.ylabel('CO2'); plt.title('Mauna Loa weekly CO2 series'); plt.tight_layout(); plt.savefig('assets/03_data_or_model.png',dpi=150); plt.close()
plt.figure(figsize=(9,5)); plt.plot(truth,label='observed'); plt.plot(p,label='forecast'); plt.xlabel('Held-out step'); plt.ylabel('CO2'); plt.title('LSTM chronological hold-out forecast'); plt.legend(); plt.tight_layout(); plt.savefig('assets/04_evaluation_or_results.png',dpi=150); plt.close(); print(json.dumps(out,indent=2))