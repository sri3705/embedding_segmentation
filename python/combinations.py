from multiprocessing import Pool
import os


def run_command(cmd):
    print cmd
    os.system(cmd)
    return cmd

machine = 'cs-vml-23'
parameters = {
        '-v': ['angkor_wat', 'animal_chase', 'arctic_kayak', 'freight_train', 'chameleons'],
        '-b': [128 ],
        '-a': [12],
        '-A': [6],
        '-S': [400,1500],
        '-o': [128],
        }

log_path = '/cs/vml2/smuralid/projects/logs/{}/'.format(machine)
proj_path = '$proj'
os.system('mkdir -p ' + log_path)
def dfs(i, params_list, cmd, processes):
    if i == len(params_list):
        params_str = ' '.join(map(str, cmd))
        comment_str = '_'.join(map(str, cmd))
        processes.append('python {0}/exec_scripts.py {1} -c {2} -f > {3}/exp_{2}_{4}.log 2>&1'.format(proj_path,params_str, comment_str, log_path, machine))
        # processes.append('python {0}/exec_scripts.py {1} -c {2} -f'.format(proj_path,params_str, comment_str))
    else:
        for p in params_list[i][1]:
            cmd.append(params_list[i][0])
            cmd.append(p)
            dfs(i+1, params_list, cmd, processes)
            cmd.pop()
            cmd.pop()
def main():
    params_list = list(parameters.iteritems())
    matlab_list = []
    processes_list = []
    dfs(0, params_list, [], processes_list)
    # print '\n'.join(processes_list)

    runner_par = run_command
    # pool = Pool()
    # pool.map(runner_par, processes_list)
    # pool.close()
    # pool.join()
    for cmd in processes_list:
        run_command(cmd)

    start = 118
    for i in xrange(len(params_list)):
        print 'run_khoreva', start+i
        os.system('python '+proj_path+'/run_khoreva.py '+str(start+i)+ ' {}/exp_{}.log 2>&1'.format(log_path, i+start))

if __name__ == '__main__':
    main()


