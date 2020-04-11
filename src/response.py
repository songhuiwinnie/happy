from geograpy import extraction
import datefinder


def get_response(user_in, seq2seq_models, wv_model):
    return seq2seq_models.answer(user_in, wv_model=wv_model)


def find_dates(message):
    available_dates, flag = datefinder.find_dates(message), True
    date_departure, date_return = None, None
    for flight_date in available_dates:
        flight_date = flight_date.strftime("%m-%d-%y")
        if flag:
            flag, departure_date = False, flight_date
        else:
            flag, return_date = True, flight_date

    return date_departure, date_return


def find_places(message):
    extractor, flag = extraction.Extractor(text=message), True
    extractor.find_entities()
    available_places, place_from, place_to = extractor.places, None, None
    for place in available_places:
        if flag:
            flag, place_from = False, place
        else:
            flag, place_to = True, place

    return place_from, place_to

