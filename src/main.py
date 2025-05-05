from params import *
from LightCard import LightCardServer, LightCardLocal


def main():

    if LOCAL_TEST:
        model_paths = "Tinker_Edge_T_models/model_F1.{}__results.pkl_class{}.pkl"
        test_paths = "test_sets/F1.{}__results.pkl_class{}_test.npy"
        LC_times_path = "/results/LC_times_E1.{}_C{}.csv"
        
        local = LightCardLocal(NUM_MODELS, NUM_CLASSES)
        local.load_paths(model_paths, test_paths, LC_times_path)
        local.run()

    else:
        # Initialize the server
        STORAGE_FILE = "/results/LC_times_E{}.{}_C{}.csv"
        STORAGE_FILE = STORAGE_FILE.format(NUM_EXP, NUM_TEST, NUM_CLASS)
        server = LightCardServer(IP_HOST, PORT_NUM, STORAGE_FILE)
        # Start the server
        server.start()

        # Load model
        model_path = "models/model_F1.{}__results.pkl_class{}.pkl"
        server.load_model(model_path.format(NUM_EXP, NUM_CLASS))

        # Handle client connections
        server.handle_client()

        # Close the server
        server.close()



if __name__ == "__main__":
    main()

