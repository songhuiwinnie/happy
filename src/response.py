from geograpy import extraction
import datefinder


COUNTRIES = ["New York", "Cambodia", "Hong Kong", "India", "Japan", "Korea", "Laos", "Myanmar", "Singapore", "Thailand",
             "Vietnam"]


def get_response(user_in, seq2seq_models, wv_model):
    return seq2seq_models.answer(user_in, wv_model=wv_model)


def find_dates(message):
    available_dates, flag = datefinder.find_dates(message), True
    date_departure, date_return = None, None

    for available_date in available_dates:
        flight_date = available_date.strftime("%m-%d-%y")
        if flag:
            flag, date_departure = False, flight_date
        else:
            flag, date_return = True, flight_date

    return date_departure, date_return


def find_places(message):
    extractor, flag = extraction.Extractor(text=message), True
    extractor.find_entities()
    available_places, place_from, place_to = extractor.places, None, None
    for place in available_places:
        if flag:
            flag, place_from = False, place.title()
        else:
            flag, place_to = True, place.title()

    return place_from, place_to

