import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from environment import *
from train import *

def draw_process(xs, ys, config):
    with PdfPages('./figs/process_N_%d_M_%d.pdf' % (config['n'], config['m'])) as pdf:
        for idx, methods_name in enumerate(config['methods']):
            plt.plot(xs[idx], ys[idx], label=methods_name, color=config['colors'][idx], linestyle=config['linestyles'][idx])
        plt.xlabel('Iteration index', fontsize=14)
        plt.ylabel('Total delay(s)', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(loc='upper right', fontsize=14)
        pdf.savefig()
        plt.close()
        # plt.show()

def draw_conver(xs, ys, config):
    with PdfPages(config['file_name'] % config['m']) as pdf:
        for idx, methods_name in enumerate(config['methods']):
            plt.plot(xs[idx], ys[idx], label=methods_name, marker=config['markers'][idx], color=config['colors'][idx], linestyle=config['linestyles'][idx])
        plt.xlabel(config['xlabel'])
        plt.ylabel(config['ylabel'])
        plt.legend(loc='upper left')
        plt.grid(ls=':', color='gray')  # 设置网格
        pdf.savefig()
        plt.close()
        # plt.show()

if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    ex_name = EX_NAME
    ex_config = EX_CONFIG[ex_name]
    if ex_name == 'cover_process':
        xs = []
        ys = []
        env = set_env(ex_config)
        for method_name in ex_config['methods']:
            e = copy.deepcopy(env)
            process, _ = train(
                e, method_name, config=ex_config['method_config'], conver=False)
            xs.append(range(ex_config['method_config']['max_epoches']))
            ys.append(process)
        draw_process(xs, ys, ex_config)
    elif ex_name == 'cover_md':
        for n in ex_config['ns']:
            xs = []
            ys = []
            env_config = {
                'n': n,
                'm': ex_config['m']
            }
            env = set_env(env_config)
            for idx, method_name in enumerate(ex_config['methods']):
                e = copy.deepcopy(env)
                process, conver_epoch = train(
                    e, method_name, config=ex_config['method_config'], conver=False)
                ys.append(process)
            xs = [range(len(process)), range(len(process))]
            ex_config['n'] = n
            draw_process(xs, ys, ex_config)
                
