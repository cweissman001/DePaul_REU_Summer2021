import pandas as pd

path = 'C:/Users/yhou14/Research/FashionMNIST/25label/'


t_sh =pd.read_csv(path+'y_pred_tr_tshirt+shirt.csv')
t_dr =pd.read_csv(path+'y_pred_tr_tshirt+dress.csv')
sh_ct =pd.read_csv(path+'y_pred_tr_shirt+coat.csv')
sh_po =pd.read_csv(path+'y_pred_tr_shirt+pullover.csv')
sh_dr =pd.read_csv(path+'y_pred_tr_shirt+dress.csv')
ct_po =pd.read_csv(path+'y_pred_tr_coat+pullover.csv')

df = pd.concat([t_sh, t_dr, sh_ct, sh_po, sh_dr, ct_po], axis=0)

import matplotlib.pyplot as plt
plt.hist(df, bins=100)
plt.grid(axis='y', alpha=0.25)
plt.title("Distribution of y_pred for pariwise similarity = 0.25")
plt.vlines(x = 0.255432, ymin = 0, ymax = 4000,
           colors = 'purple',
           label = 'vline_multiple - full height')
plt.vlines(x = 0.892788, ymin = 0, ymax = 4000,
           colors = 'purple',
           label = 'vline_multiple - full height')
plt.show()






