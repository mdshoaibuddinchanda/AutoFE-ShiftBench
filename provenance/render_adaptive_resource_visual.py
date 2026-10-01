"""Render frozen CPU/RAM/VRAM budgets, not workload utilization."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def main():
    scope_path=ROOT/'provenance'/'reviewer1_launch_scope_v4.json'
    plan=json.loads(scope_path.read_text())['resource_plan']
    hardware=plan['hardware'];settings=plan['settings']
    gpu_total=sum(d['total_bytes'] for d in hardware['gpu_devices'])/1024**3
    values=[('Logical CPU slots',len(hardware['allowed_cpu_ids']),plan['worker_ceiling']),
            ('RAM (GiB)',hardware['ram_total_bytes']/1024**3,plan['ram_budget_bytes']/1024**3),
            ('VRAM (GiB)',gpu_total,gpu_total*settings['vram_target_fraction'])]
    fig,axes=plt.subplots(1,3,figsize=(11,3.8))
    for ax,(title,total,budget) in zip(axes,values):
        ax.barh([0],[budget],color='#286fb0',label='Worker ceiling / memory budget')
        ax.barh([0],[total-budget],left=[budget],color='#d8e1ed',label='Reserved headroom')
        ax.set_xlim(0,total*1.05);ax.set_ylim(-.8,.8);ax.set_yticks([])
        ax.set_title(title,fontsize=11)
        ax.text(budget/2,0,f'{budget:.2f}' if title!='Logical CPU slots' else str(int(budget)),
                color='white',ha='center',va='center',weight='bold')
        ax.text(budget+(total-budget)/2,0,f'{total-budget:.2f}' if title!='Logical CPU slots' else str(int(total-budget)),
                ha='center',va='center',fontsize=9)
        ax.set_xlabel(f'Total: {total:.2f}' if title!='Logical CPU slots' else f'Total: {int(total)}')
        ax.spines[['top','right','left']].set_visible(False)
    fig.suptitle('Adaptive host budgets — execution policy, not measured utilization',fontsize=13,y=.96)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncols=2,fontsize=8)
    fig.subplots_adjust(left=.035,right=.985,bottom=.26,top=.76,wspace=.20)
    output=ROOT/'provenance'/'figures';output.mkdir(exist_ok=True)
    for extension in ('png','pdf'):fig.savefig(output/f'adaptive_resource_budget.{extension}',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
