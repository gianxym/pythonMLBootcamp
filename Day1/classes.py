import math

# This class is a collection of variables and functions that can describe automobiles
class Automobile:


    # This is a special function and is a Constructor: It only runs per object, when it is created
    def __init__(self, gas_capacity, max_speed, euros_per_km):

        # Constraining Variables
        self.gas_capacity = gas_capacity
        self.max_speed = max_speed
        self.euros_per_km = euros_per_km   # How much it costs to travel 1 km

        # Current state of automobile (Default values )
        self.gas = 0
        self.speed = 0   # kms / h
        self.position = 0

        # Automobile's other information
        self.model = None
        self.weight = None


    # A function for filling gas
    def fill_gas(self, euros):

        # Do not allow negative values of euros
        if (euros < 0):
            print("Negative money is not allowed")
            return

        # Calculate how much gas you get for the money you give at current gas prices of 1.88 euros per liter
        liters = euros / 1.88

        if (self.gas + liters > self.gas_capacity):
            print("Added " + str(self.gas_capacity - self.gas) + " liters")
            print("Gas tank at: 100%")
            print("Only used " + (str(round(1.88 * (self.gas_capacity - self.gas), 3)))+ " / " + str(euros) + " Euros.")
            self.gas = self.gas_capacity
        else:
            self.gas += liters
            print("Added " + str(liters) + " liters")
            print("Gas tank at: ", 100 * self.gas / self.gas_capacity, "%", sep="")


    # Function that sets car's speed
    def set_speed(self, speed):
        if speed > self.max_speed:
            self.speed = self.max_speed
            print("Speed set to " + str(self.max_speed) + " km/h")
        else:
            self.speed = speed
            print("Speed set to " + str(speed) + " km/h")


    # Function for traveling
    def travel(self, hours):
        
        # Calculate new position:
        self.position += hours * self.speed

        # Calculate gas lost
        gas_spend = self.euros_per_km / 1.88 * hours * self.speed

        if gas_spend > self.gas:
            true_hours = self.gas * 1.88 / self.euros_per_km / self.speed
            self.gas = 0
            print("You ran out of gas at ", round(true_hours,2), " hours and traveled ", round(true_hours * self.speed,3), " kilometers", sep="")
        else:
            self.gas -= gas_spend
            print("You traveled ", round(hours * self.speed,3), " kilometers", sep="")


    def __str__(self):
        return f"{self.model} (weight: {self.weight} kg)\n" \
               f"Gas: {self.current_gas:.2f}/{self.max_gas_capacity:.2f} liters\n" \
               f"Speed: {self.current_speed} km/h\n" \
               f"Position: {self.current_position}"
    



# Try testing the classes

# Motorbike
motorbike = Automobile(gas_capacity=10, max_speed=120, euros_per_km=0.1)
motorbike.model = "Honda CBR"
motorbike.weight = 196  # kg

# SUV object
suv = Automobile(gas_capacity=80, max_speed=180, euros_per_km=0.3)
suv.model = "Toyota RAV4"
suv.weight = 1700

# Truck object
truck = Automobile(gas_capacity=200, max_speed=120, euros_per_km=0.5)
truck.model = "Volvo VNL 860"
truck.weight = 9000


# Play around with motorbike
# Try to pay 100 euros to fill gas
motorbike.fill_gas(100)
print()

# Set speed at 50 km / hour
motorbike.set_speed(50)
print()

# Try to travel 20 km
motorbike.travel(20)
print()

# Try to pay 100 euros to fill gas
motorbike.fill_gas(100)
print()

# Try to set speed to 150 km / hour
motorbike.set_speed(150)
print()

# Try to travel 20 km
motorbike.travel(20)
print()
