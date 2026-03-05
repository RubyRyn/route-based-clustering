from ClientWardsMapping import WardMapper

# Step 1: Load wards and map clients
mapper = WardMapper(geojson_path="Input\\Ayeyarwady.json")
client_ward_map = mapper.map_clients(locations)
# Fix any clients that fell outside ward boundaries
mapper.fix_unmapped_clients(locations)

# See the results
mapper.summary()

# These will be useful for Step 2:
ward_clients = mapper.get_ward_client_map()    # ward_pcode -> [client indices]
client_wards = mapper.get_client_ward_labels()  # client_idx -> ward_pcode