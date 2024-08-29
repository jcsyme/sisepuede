import warnings
warnings.filterwarnings("ignore")
import argparse
import model_attributes as ma
from attribute_table import AttributeTable
import os, os.path
import numpy as np
import pandas as pd
import sector_models as sm
import setup_analysis as sa
import support_functions as sf
import sqlalchemy






def parse_arguments() -> dict:

    parser = argparse.ArgumentParser(description = "Run SISEPUEDE models from the command line.")
    parser.add_argument(
        "--input",
        type = str,
        help = f"Path to an input CSV, long by {sa.model_attributes.dim_time_period}, that contains required input variables."
    )
    parser.add_argument(
        "--output",
        type = str,
        help = "Path to output csv file",
        default = sa.fp_csv_default_single_run_out
    )
    parser.add_argument(
        "--models",
        type = str,
        help = "Models to run using the input file. Possible values include 'All' (run all models) or any comma-delimited combination of the following: 'AFOLU', 'CircularEconomy', 'ElectricEnergy', 'IPPU', and 'NonElectricEnergy'",
        default = "All"
    )
    parser.add_argument(
        "--integrated",
        help = "Include this flag to run included models as integrated sectors. Output from upstream models will be passed as inputs to downstream models.",
        action = "store_true"
    )
    parsed_args = parser.parse_args()

    # Since defaults are env vars, still need to checking to make sure its passed
    errors = []
    if parsed_args.input is None:
        errors.append("Missing --input DATA INPUT FILE")
    if errors:
        raise ValueError(f"Missing arguments detected: {sf.format_print_list(errors)}")

    # json args over-write specified args
    parsed_args_as_dict = vars(parsed_args)

    return parsed_args_as_dict



def main(args: dict) -> None:

    print("\n***\n***\n*** Bienvenidos a SISEPUEDE! Hola Edmundo y equipo ITEM—esta mensaje va a cambiar en el futuro, y hoy pasa este dia. Mira, incluye electricidad, que rico. Espero que todavia disfruten esta mensaje *cada vez*.\n***\n***\n")

    fp_in = args.get("input")
    fp_out = args.get("output")
    models_run = args.get("models").split(",")
    models_run = models_run if (models_run[0].lower() != "all") else ["AFOLU", "CircularEconomy", "ElectricEnergy", "IPPU", "NonElectricEnergy"]


    # load data
    if not fp_in:
        raise ValueError("Cannot run: no input data file was specified.")
    else:
        if os.path.exists(args["input"]):
            print(f"Reading input data from {fp_in}...")
            df_input_data = pd.read_csv(fp_in)
            print("Done.")
        else:
            raise ValueError(f"Input file '{fp_in}' not found.")

    # notify of output path
    print(f"\n\n*** STARTING MODELS ***\n\nOutput file will be written to {fp_out}.\n")

    init_merge_q = True
    run_integrated_q = bool(args.get("integrated"))
    df_output_data = []


    ##  RUN MODELS

    # run AFOLU and collect output
    if "AFOLU" in models_run:
        print("\n\tRunning AFOLU")
        # get the model, run it using the input data, then update the output data (for integration)
        model_afolu = sm.AFOLU(sa.model_attributes)
        df_output_data.append(model_afolu.project(df_input_data))


    # run CircularEconomy and collect output
    if "CircularEconomy" in models_run:
        print("\n\tRunning CircularEconomy")
        model_circecon = sm.CircularEconomy(sa.model_attributes)
        # integrate AFOLU output?
        if run_integrated_q and set(["AFOLU"]).issubset(set(models_run)):
            df_input_data = sa.model_attributes.transfer_df_variables(
                df_input_data,
                df_output_data[0],
                model_circecon.integration_variables
            )
        df_output_data.append(model_circecon.project(df_input_data))
        df_output_data = [sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")] if run_integrated_q else df_output_data


    # run IPPU and collect output
    if "IPPU" in models_run:
        print("\n\tRunning IPPU")
        model_ippu = sm.IPPU(sa.model_attributes)
        # integrate Circular Economy output?
        if run_integrated_q and set(["CircularEconomy"]).issubset(set(models_run)):
            df_input_data = sa.model_attributes.transfer_df_variables(
                df_input_data,
                df_output_data[0],
                model_ippu.integration_variables
            )
        df_output_data.append(model_ippu.project(df_input_data))
        df_output_data = [sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")] if run_integrated_q else df_output_data


    # run Non-Electric Energy and collect output
    if "NonElectricEnergy" in models_run:
        print("\n\tRunning NonElectricEnergy")
        model_energy = sm.NonElectricEnergy(sa.model_attributes)
        # integrate IPPU output?
        if run_integrated_q and set(["IPPU", "AFOLU"]).issubset(set(models_run)):
            df_input_data = sa.model_attributes.transfer_df_variables(
                df_input_data,
                df_output_data[0],
                model_energy.integration_variables_non_fgtv
            )
        else:
            print("LOG ERROR HERE: CANNOT RUN WITHOUT IPPU")
        df_output_data.append(model_energy.project(df_input_data))
        df_output_data = [sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")] if run_integrated_q else df_output_data


    # run Electricity and collect output
    if "ElectricEnergy" in models_run:
        print("\n\tRunning ElectricEnergy")
        model_elecricity = sm.ElectricEnergy(sa.model_attributes, sa.dir_ref_nemo)
        # integrate energy-related output?
        if run_integrated_q and set(["CircularEconomy", "AFOLU", "NonElectricEnergy"]).issubset(set(models_run)):
            df_input_data = sa.model_attributes.transfer_df_variables(
                df_input_data,
                df_output_data[0],
                model_elecricity.integration_variables
            )

        # create the engine
        engine = sqlalchemy.create_engine(f"sqlite:///{sa.fp_sqlite_nemomod_db_tmp}")
        try:
            df_elec =  model_elecricity.project(df_input_data, engine)
            df_output_data.append(df_elec)
            df_output_data = [sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")] if run_integrated_q else df_output_data
        except Exception as e:
            #LOGGING
            print(f"Error running ElectricEnergy model: {e}")


    # finally, add fugitive emissions from Non-Electric Energy and collect output
    if "NonElectricEnergy" in models_run:
        print("\n\tRunning NonElectricEnergy - Fugitive Emissions")
        model_energy = sm.NonElectricEnergy(sa.model_attributes)
        # integrate IPPU output?
        if run_integrated_q and set(["IPPU", "AFOLU"]).issubset(set(models_run)):
            df_input_data = sa.model_attributes.transfer_df_variables(
                df_input_data,
                df_output_data[0],
                model_energy.integration_variables_fgtv
            )
        else:
            print("LOG ERROR HERE: CANNOT RUN WITHOUT IPPU AND AFOLU")

        df_output_data.append(model_energy.project(df_input_data, subsectors_project = sa.model_attributes.subsec_name_fgtv))
        df_output_data = [sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")] if run_integrated_q else df_output_data


    # build output data frame
    df_output_data = sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")


    print("\n*** MODEL RUNS COMPLETE ***\n")

    # write output
    print(f"\nWriting data to {fp_out}...")
    df_output_data.to_csv(fp_out, index = None, encoding = "UTF-8")
    print("\n*** MODEL RUNS SUCCESSFULLY COMPLETED. Q les vayan bien damas y caballeros.")


if __name__ == "__main__":

    args = parse_arguments()

    main(args)
