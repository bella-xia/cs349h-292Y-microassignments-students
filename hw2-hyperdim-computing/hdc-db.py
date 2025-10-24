from hdc import *
import csv


class HDDatabase:

    def __init__(self):
        self.db = HDItemMem("db")
        self.cb = HDCodebook()
        self.field_ks = None
        return
        
        # unreachable
        raise Exception("other instantiations here")

    def encode_string(self, value : str) -> np.ndarray:
        if not self.cb.has(value):
            self.cb.add(value) 
        return self.cb.get(value)

        # unreachable
        raise Exception("translate a string to a hypervector")

    def decode_string(self, hypervec : np.ndarray) -> str:
        approx_str, _ = self.cb.wta(hypervec)
        return approx_str

        # unreachable
        raise Exception("translate a hypervector to a string")

    def encode_row(self, fields : dict[str, str]) -> np.ndarray:
        if not self.field_ks:
            self.field_ks = list(fields.keys())

        xs : list[np.ndarray] = []
        for k, v in fields.items():
            k_vec : np.ndarray = self.encode_string(k)
            v_vec : np.ndarray = self.encode_string(v)
            xs.append(HDC.bind(k_vec, v_vec))
        return HDC.bundle(xs)
        
        # unreachable
        raise Exception("translate a dictionary of field-value pairs to a hypervector")

    def decode_row(self, hypervec : np.ndarray) -> dict[str, str]:
        decoded_fields : dict[str, str] = {}

        for k in self.field_ks:
            k_vec : np.ndarray = self.encode_string(k)
            approx_vec : np.ndarray = HDC.bind(hypervec, k_vec)
            v : str = self.decode_string(approx_vec)
            decoded_fields[k] = v

        return decoded_fields

        # unreachable
        raise Exception("reconstruct a dictionary of field-value pairs from a hypervector.")

    def add_row(self, primary_key, fields) -> None:
        fields_vec : np.ndarray = self.encode_row(fields)
        self.db.add(primary_key, fields_vec)
        return

        # unreachable
        raise Exception("add a database row.")

    def get_row(self, key):
        return self.decode_row(self.db.get(key))

        # unreachable
        raise Exception("retrieve a dictonary of field-value pairs from a hypervector row")

    def get_value(self, key, field):
        row_vec : np.ndarray = self.db.get(key)
        field_vec : np.ndarray = self.encode_string(field)
        approx_value_vec : np.ndarray = HDC.bind(field_vec, row_vec)
        return self.decode_string(approx_value_vec)

        # unreachable
        raise Exception("given a primary key and a field, get the value assigned to the field")

    def get_matches(self, field_value_dict, threshold):
        row_vec : np.ndarray = self.encode_row(field_value_dict)
        matches : dict[str, float] =  self.db.matches(row_vec, threshold=threshold)
        return matches
        
        # unreachable
        raise Exception("get database entries that contain provided dictionary of field-value pairs")

    def get_analogy(self, target_key, other_key, target_value):
        target_vec : np.ndarray = self.db.get(target_key)
        other_vec : np.ndarray = self.db.get(other_key)
        value_vec : np.ndarray = self.encode_string(target_value)
        approx_field_vec : np.ndarray = HDC.bind(value_vec, target_vec)
        other_value_vec : np.ndarray = HDC.bind(approx_field_vec, other_vec)

        return self.cb.wta(other_value_vec)

        # unreachable
        raise Exception("analogy query")


def load_json():
    data = {}
    with open("digimon.csv", "r") as csvf:
        csvReader = csv.DictReader(csvf)
        for rows in csvReader:
            key = rows['Digimon']
            data[key] = rows
    return data


def build_database(data):
    HDC.SIZE = 10000
    db = HDDatabase()

    for key, fields in data.items():
        db.add_row(key, fields)

    return db


def summarize_result(data, result, summary_fn):
    print("---- # matches = %d ----" % len(list(result.keys())))
    for digi, distance in result.items():
        print("%f] %s: %s" % (distance, digi, summary_fn(data[digi])))


def digimon_basic_queries(data, db):

    print("===== virus-plant query =====")
    thr = 0.40
    digis = db.get_matches({"Type": "Virus", "Attribute": "Plant"}, threshold=thr)
    summarize_result(data, digis, lambda row: "true match" if row["Type"] == "Virus" and row["Attribute"] == "Plant" else "false positive")

    print("===== champion query =====")
    thr = 0.40
    digis = db.get_matches({"Stage": "Champion"}, threshold=thr)
    summarize_result(data, digis, lambda row: "true match" if row["Stage"] == "Champion" else "false positive")


def digimon_test_encoding(data, db):
    strn = "tester"
    hv_test = db.encode_string(strn)
    rec_strn = db.decode_string(hv_test)
    print("original=%s" % strn)
    print("recovered=%s" % rec_strn)
    print("---")

    row = data["Wormmon"]
    hvect = db.encode_row(row)
    rec_row = db.decode_row(hvect)
    print("original=%s" % str(row))
    print("recovered=%s" % str(rec_row))
    print("---")


def digimon_value_queries(data, db):
    value = db.get_value("Lotosmon", "Stage")
    print("Lotosmon.Stage = %s" % value)

    targ_row = db.get_row("Lotosmon")
    print("Lotosmon" + str(targ_row))


def analogy_query(data, db):
    # Lotosmon is to Data as Imperialdramon PM is to <what field>

    targ_row = db.get_row("Lotosmon")
    other_row = db.get_row("Imperialdramon PM")
    print("Lotosmon has a a field with a Data value, what is the equivalent value in Imperialdramon PM's entry")
    value, dist = db.get_analogy(target_key="Lotosmon", other_key="Imperialdramon PM", target_value="Data")
    print("Lotosmon" + str(targ_row))
    print("Imperialdramon PM" + str(other_row))
    print("------")
    print("value: %s (dist=%f)" % (value, dist))
    print("expected result: Vaccine, the type of Imperialdramon PM")
    print("")


if __name__ == '__main__':
    data = load_json()
    db = build_database(data)
    digimon_test_encoding(data, db)
    digimon_basic_queries(data, db)
    digimon_value_queries(data, db)
    analogy_query(data, db)