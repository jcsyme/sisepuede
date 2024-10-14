

#########################
#    SET DIRECTORIES    #
#########################

# load support functions
include("support_functions.jl")


####################
#    STRUCTURES    #
####################

struct AttributeTable

    fp_table::String
    key::Symbol
    fields_to_dict::Array{Symbol, 1}
    clean_table_fields::Bool
    table
    field_maps
    key_vals

    function AttributeTable(fp_table::String, key::Symbol, fields_to_dict::Array{Symbol, 1}, clean_table_fields::Bool)

        # verify table exists and check keys
        table = read_csv(check_path(fp_table, false), true)
        fields_to_dict = [x for x in fields_to_dict if x != key]
        check_fields!(table, [[key]; fields_to_dict])

        # check key
        if length(Set(table[:, key])) < nrow(table)
            error("Invalid key '$(key)' found in $(fp_table): the key is not unique. Check the table and specify a unique key.")
        end

        # clean the keys?
        if clean_table_fields
            clean_field_names!(table)
            fields_to_dict = clean_field_names(fields_to_dict)
            key = clean_field_names([key])[1]
        end

        # clear RST formatting
        table[!, key] = replace.(replace.(table[!, key], "`" => ""), "\$" => "")

        # next, create dict maps
        field_maps = Dict()
        for fld in fields_to_dict
            field_fwd = "$(key)_to_$(fld)"
            field_rev = "$(fld)_to_$(key)"

            field_maps[field_fwd] = build_dict(table[:, [key, fld]])
            # check for 1:1 correspondence before adding reverse
            vals_unique = Set(table[:, fld])
            if (length(vals_unique) == nrow(table))
                field_maps[field_rev] = build_dict(table[:, [fld, key]])
            end
        end

        # get all values associated with they key
        key_vals = sort(collect(Set(table[:, key])))

        return new(fp_table, key, fields_to_dict, clean_table_fields, table, field_maps, key_vals)
    end
end



## high level directory structure
dir_proj = dirname(dirname(@__FILE__))
#dir_data = check_path(joinpath(dirname(dir_proj), "data"), false)
dir_bin = joinpath(dir_proj, "bin")
dir_out = check_path(joinpath(dir_proj, "out"), true)
dir_py = check_path(joinpath(dir_proj, "python"), false)
dir_ref = check_path(joinpath(dir_proj, "ref"), false)
dir_tmp = joinpath(dir_proj, "tmp")

# file path for the nemomod database
fp_sqlite_nemomod_db_tmp = joinpath(dir_tmp, "nemomod_intermediate_database.sqlite")
