from veros.settings import Setting

SETTINGS = {
    "start_date": Setting("",str, "Start date of simulation"),
    "dt_forcing": Setting(3, int, "Time step for forcing in hours"),
    "smooth": Setting(2, int, "scale ratio for topography gaussian filtering"),
    }
