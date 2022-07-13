from importlib import import_module
def converging(generate_file, truth_file, converge_method_name):


    mod = import_module(converge_method_name +'.method')
    em = getattr(mod, converge_method_name)(generate_file, truth_file)
    em.run()
    accuracy = em.get_accuracy()
    return accuracy
