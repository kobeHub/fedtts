from multiprocessing import Pool
import subprocess

def worker(args):
    task, model, dataset, local_ep, epochs, target_accuracy, eval_every, local_bs, frac, eval_after, local_algo, n_cluster, r_overlapping, gamma, n_transfer, config_file, eid, verbose = args
    result = subprocess.run(['python', 'your_python_script.py',
                             '--model', model,
                             '--dataset', dataset,
                             '--local_ep', str(local_ep),
                             '--epochs', str(epochs),
                             '--target_accuracy', str(target_accuracy),
                             '--eval_every', str(eval_every),
                             '--local_bs', str(local_bs),
                             '--frac', str(frac),
                             '--eval_after', str(eval_after),
                             '--local_algo', local_algo,
                             '--n_cluster', str(n_cluster),
                             '--r_overlapping', str(r_overlapping),
                             '--gamma', str(gamma),
                             '--n_transfer', str(n_transfer),
                             '--config_file', config_file,
                             '--eid', str(eid),
                             '--verbose', str(verbose)],
                             capture_output=True, text=True)
    
    # Write the result to a file named output_taskX.txt, where X is the task number
    with open(f'output_task{task}.txt', 'w') as file:
        file.write(result.stdout)

if __name__ == '__main__':
    # Define the number of cores
    num_cores = 8

    # Define the tasks and additional arguments as tuples
    tasks_and_args = [
        (1, 'cnn', 'cifar10', 5, 1000, 0.96, 2, 10, 0.1, 100, 'FedAvg', 3, 0, 0.1, 5, './config/fedtts-conf.yaml', 20, 0),
        # Add more tuples for additional tasks
        (2, 'cnn', 'cifar10', 5, 1000, 0.96, 2, 10, 0.1, 100, 'FedTTS', 3, 0, 0.1, 5, './config/fedtts-conf.yaml', 20, 0),
        # ... Repeat for 6 more tuples
        (3, 'cnn', 'cifar10', 5, 1000, 0.96, 2, 10, 0.1, 100, 'FedAvg', 4, 0, 0.1, 5, './config/fedtts-conf.yaml', 21, 0),
        # Add more tuples for additional tasks
        (4, 'cnn', 'cifar10', 5, 1000, 0.96, 2, 10, 0.1, 100, 'FedTTS', 4, 0, 0.1, 5, './config/fedtts-conf.yaml', 21, 0),
        (5, 'cnn', 'cifar10', 5, 1000, 0.96, 2, 10, 0.1, 100, 'FedAvg', 5, 0, 0.1, 5, './config/fedtts-conf.yaml', 22, 0),
        # Add more tuples for additional tasks
        (6, 'cnn', 'cifar10', 5, 1000, 0.96, 2, 10, 0.1, 100, 'FedTTS', 5, 0, 0.1, 5, './config/fedtts-conf.yaml', 22, 0),
        (7, 'cnn', 'cifar10', 5, 1000, 0.96, 2, 10, 0.1, 100, 'FedAvg', 3, 0.1, 0.1, 5, './config/fedtts-conf.yaml', 23, 0),
        # Add more tuples for additional tasks
        (8, 'cnn', 'cifar10', 5, 1000, 0.96, 2, 10, 0.1, 100, 'FedTTS', 3, 0.1, 0.1, 5, './config/fedtts-conf.yaml', 23, 0),
    ]

    # Create a Pool with the specified number of cores
    with Pool(processes=num_cores) as pool:
        # Map the worker function to the tasks and additional arguments, distributing them across the Pool
        pool.map(worker, tasks_and_args)

