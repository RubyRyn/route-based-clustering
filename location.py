class Location:
    """Represents a location (office or client)"""
    def __init__(self, id: str, name: str, lat: float, lon: float, loc_type: str):
        self.id = id
        self.name = name
        self.lat = lat
        self.lon = lon
        self.type = loc_type
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'lat': self.lat,
            'lon': self.lon,
            'type': self.type
        }