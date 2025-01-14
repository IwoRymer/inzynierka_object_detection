def save_label(cars, people, movement, filename):
    """
    Saves the variables cars, people, and movement to a text file.
    """
    with open(filename, 'w') as file:
        file.write(f"cars={cars}\n")
        file.write(f"people={people}\n")
        file.write(f"movement={movement}\n")


def read_label(filename):
    """
    Reads the variables from a text file and returns them as a dictionary.
    """
    variables = {}
    with open(filename, 'r') as file:
        for line in file:
            key, value = line.strip().split("=")
            # Convert the value to int or float if possible, otherwise keep as string
            try:
                variables[key] = int(value)
            except ValueError:
                try:
                    variables[key] = float(value)
                except ValueError:
                    variables[key] = value
    return variables

def label_photo(image_number):
    people = int(input("Note down number of people: "))
    cars = int(input("Note down number of cars on the picture: "))
    movement = int(input("Note if target is moving, 0 = No, >0 = Yes: "))
    label_filename = f"label_{image_number}.txt"
    save_label(cars, people, movement, label_filename)


def label_photo_return(image_number):
    people = int(input("Note down number of people: "))
    cars = int(input("Note down number of cars on the picture: "))
    movement = int(input("Note if target is moving, 0 = No, >0 = Yes: "))
    label_filename = f"label_{image_number}.txt"
    save_label(cars, people, movement, label_filename)
    return cars, people, movement

def read_label_return(filename):
    loaded_variables = read_label(filename)
    cars = loaded_variables['cars']
    people = loaded_variables['people']
    mov = loaded_variables['movement']
    return cars, people, mov




    


# Example usage
#i, j, k = 10, 20, 30  # Example values
#save_label(i, j, k, "variables.txt")
#cars, people, mov = read_label_return("variables.txt")
#print(cars, people, mov)
#loaded_variables = read_from_file("variables.txt")
#print(loaded_variables['cars'], loaded_variables['people'], loaded_variables['movement'])  # Output: {'cars': 10, 'people': 20, 'movement': 30}