class Victim:
    def __init__(self, date, name, age, race, cause, neighbourhood, time, address):
        self.date = date
        self.name = name
        self.age = age
        self.race = race
        self.cause = cause
        self.neighbourhood = neighbourhood
        self.time = time
        self.address = address

    def get_date(self):
        return self.date
    def get_name(self):
        return self.name
    def get_age(self):
        return self.age
    def get_race(self):
        return self.race
    def get_cause(self):
        return self.cause
    def get_neighbourhood(self):
        return self.neighbourhood
    def get_time(self):
        return self.time
    def get_address(self):
        return self.address

