import numpy as np
import os
import os.path as osp
import matplotlib.pyplot as plt
import seaborn as sns

rect_length, rect_width = 40, 30
ris_center = np.array([-12, 40, 20])

sns.set_theme(context='paper', style='whitegrid', palette='bright', font_scale=1.2)
fig_dpi = 600

fig = plt.figure()
ax = plt.gca()
# x = (np.arange(10)+1)*20

# color = next(ax._get_lines.prop_cycler)['color']
# opt = np.load('Res/algo_compare/adapt_step/loc_seq.npy')
# ax.plot(opt[::100, 1], opt[::100, 2], 'o-', color=color, label='adaptive coordinate descent step')

# color = next(ax._get_lines.prop_cycler)['color']
# ref = np.load('Res/algo_compare/fix_step/loc_seq.npy')[:opt.shape[0]]
# ax.plot(ref[::100, 1], ref[::100, 2], 'x-', color=color, label='fixed coordinate descent step')

ax.plot([ris_center[1]-rect_length/2, ris_center[1]-rect_length/2],\
    [ris_center[2]+rect_width/2, ris_center[2]-rect_width/2], 'g-')
ax.plot([ris_center[1]-rect_length/2, ris_center[1]+rect_length/2],\
    [ris_center[2]+rect_width/2, ris_center[2]+rect_width/2], 'g-')
ax.plot([ris_center[1]-rect_length/2, ris_center[1]+rect_length/2],\
    [ris_center[2]-rect_width/2, ris_center[2]-rect_width/2], 'g-')
ax.plot([ris_center[1]+rect_length/2, ris_center[1]+rect_length/2],\
    [ris_center[2]-rect_width/2, ris_center[2]+rect_width/2], 'g-')

ax.plot(ris_center[1]-rect_length/2, ris_center[2]+rect_width/2, 'b.', markersize=15)
ax.plot(ris_center[1]-rect_length/2, ris_center[2]-rect_width/2, 'b.', markersize=15)
ax.plot(ris_center[1]+rect_length/2, ris_center[2]-rect_width/2, 'b.', markersize=15)
ax.plot(ris_center[1]+rect_length/2, ris_center[2]+rect_width/2, 'b.', markersize=15)
ax.plot(ris_center[1], ris_center[2], 'b.', markersize=15)

ax.annotate('loc 0', xy=(ris_center[1]-rect_length/2, ris_center[2]+rect_width/2), \
    xytext=(ris_center[1]-rect_length/2-4, ris_center[2]+rect_width/2))
ax.annotate('loc 1', xy=(ris_center[1]+rect_length/2, ris_center[2]+rect_width/2), \
    xytext=(ris_center[1]+rect_length/2+1, ris_center[2]+rect_width/2))
ax.annotate('loc 2', xy=(ris_center[1], ris_center[2]), \
    xytext=(ris_center[1]-2, ris_center[2]-2))
ax.annotate('loc 3', xy=(ris_center[1]-rect_length/2, ris_center[2]-rect_width/2), \
    xytext=(ris_center[1]-rect_length/2-4, ris_center[2]-rect_width/2))
ax.annotate('loc 4', xy=(ris_center[1]+rect_length/2, ris_center[2]-rect_width/2), \
    xytext=(ris_center[1]+rect_length/2+1, ris_center[2]-rect_width/2))

ax.text(ris_center[1]-0.5, ris_center[2]+4, s=r'$\mathcal{B}$', fontsize=20)

ax.set_xlim([ris_center[1]-rect_length/2-5, ris_center[1]+rect_length/2+5])
ax.set_ylim([ris_center[2]-rect_width/2-5, ris_center[2]+rect_width/2+5])
ax.set_xlabel('y axis')
ax.set_ylabel('z axis')
# ax.legend(loc='best')

# plt.show()
# exit()
sns.despine()
fig.savefig(osp.join('figures', 'loc_annotate.pdf'), dpi=fig_dpi, bbox_inches='tight')

