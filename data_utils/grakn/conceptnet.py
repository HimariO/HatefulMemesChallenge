from grakn.client import GraknClient
import csv
import random
from loguru import logger

REL = [
    "/r/RelatedTo",
    "/r/FormOf",
    "/r/IsA",
    "/r/PartOf",
    "/r/HasA",
    "/r/UsedFor",
    "/r/CapableOf",
    "/r/AtLocation",
    "/r/Causes",
    "/r/HasSubevent",
    "/r/HasFirstSubevent",
    "/r/HasLastSubevent",
    "/r/HasPrerequisite",
    "/r/HasProperty",
    "/r/MotivatedByGoal",
    "/r/ObstructedBy",
    "/r/Desires",
    "/r/CreatedBy",
    "/r/Synonym",
    "/r/Antonym",
    "/r/DistinctFrom",
    "/r/DerivedFrom",
    "/r/SymbolOf",
    "/r/DefinedAs",
    "/r/MannerOf",
    "/r/LocatedNear",
    "/r/HasContext",
    "/r/SimilarTo",
    "/r/EtymologicallyRelatedTo",
    "/r/EtymologicallyDerivedFrom",
    "/r/CausesDesire",
    "/r/MadeOf",
    "/r/ReceivesAction",
    "/r/ExternalURL",
]



def company_template(company):
    return 'insert $company isa company, has name "' + company["name"] + '";'

def person_template(person):
    # insert person
    graql_insert_query = f'insert $person isa person, has phone-number "{ person["phone_number"] }"'
    if person["first_name"] == "":
        # person is not a customer
        graql_insert_query += ", has is-customer false"
    else:
        # person is a customer
        graql_insert_query += f"""
        , has is-customer true
        , has first-name "{ person["first_name"] }"
        , has last-name "{ person["last_name"] }"
        , has city "{ person["city"] }"
        , has age "{ str(person["age"]) }"
        """
    graql_insert_query += ";"
    return graql_insert_query

def contract_template(contract):
    # match company
    graql_insert_query = 'match $company isa company, has name "' + contract["company_name"] + '";'
    # match person
    graql_insert_query += ' $customer isa person, has phone-number "' + contract["person_id"] + '";'
    # insert contract
    graql_insert_query += " insert (provider: $company, customer: $customer) isa contract;"
    return graql_insert_query

def call_template(call):
    # match caller
    graql_insert_query = 'match $caller isa person, has phone-number "' + call["caller_id"] + '";'
    # match callee
    graql_insert_query += ' $callee isa person, has phone-number "' + call["callee_id"] + '";'
    # insert call
    graql_insert_query += (" insert $call(caller: $caller, callee: $callee) isa call; " +
                           "$call has started-at " + call["started_at"] + "; " +
                           "$call has duration " + str(call["duration"]) + ";")
    return graql_insert_query


def parse_data_to_dictionaries(input):
    items = []
    with open(input["data_path"] + ".csv") as data: # 1
        for row in csv.DictReader(data, skipinitialspace = True):
            item = { key: value for key, value in row.items() }
            items.append(item) # 2
    return items


def load_data_into_grakn(input, session):
    items = parse_data_to_dictionaries(input)

    for item in items:
        with session.transaction().write() as transaction:
            graql_insert_query = input["template"](item)
            print("Executing Graql Query: " + graql_insert_query)
            transaction.query(graql_insert_query)
            transaction.commit()

    print("\nInserted " + str(len(items)) + " items from [ " + input["data_path"] + "] into Grakn.\n")


def build_phone_call_graph(inputs):
    with GraknClient(uri="localhost:48555") as client:
        with client.session(keyspace = "phone_calls") as session:
            for input in inputs:
                print("Loading from [" + input["data_path"] + "] into Grakn ...")
                load_data_into_grakn(input, session)

"""
SEP
"""

def fitler_csv_by_language(csv_path, output_path):
    result_lines = []
    with open(csv_path, 'r') as f:
        for line in f:
            _, relation, entity_a, entity_b, meta = line.split('\t')
            is_en = entity_a.startswith('/c/en/') and entity_b.startswith('/c/en')
            if is_en:
                result_lines.append(line)
    
    print('result_lines: ', len(result_lines))
    with open(output_path, 'w') as f:
        f.writelines(result_lines)


def build_en_graph():
    # build_phone_call_graph(inputs=inputs)
    conceptnet_csv = '/home/ron/Downloads/hateful_meme_cache/conceptnet-assertions-5.7.0.csv'
    conceptnet_csv_en = '/home/ron/Downloads/hateful_meme_cache/conceptnet-en-5.7.0.csv'
    fitler_csv_by_language(
        conceptnet_csv,
        conceptnet_csv_en,
    )


def define_schema(host='35.234.48.188', port=48555):
    schema = """
    define

    edge sub relation,
        relates head,
        relates tail,
        has edge_id,
        has edge_type;

    node sub entity,
        plays head,
        plays tail,
        has name,
        has supply_info_json;
    
    edge_id sub attribute,
        value long;
    
    edge_type sub attribute,
        value string;
    
    name sub attribute,
        value string;
    
    supply_info_json sub attribute,
        value string;
    """
    with GraknClient(uri=f"{host}:{port}") as client:
        with client.session(keyspace = "conceptnet") as session:
            with session.transaction().write() as transaction:
                transaction.query(schema)
                transaction.commit()


def delete_all(host='35.234.48.188', port=48555):
    query = """
    match $p isa node; $r isa edge; 
    delete $p isa node; $r isa edge;
    """
    with GraknClient(uri=f"{host}:{port}") as client:
        with client.session(keyspace = "conceptnet") as session:
            with session.transaction().write() as transaction:
                transaction.query(query)
                transaction.commit()


def gql_insert_node(head_name, tail_name, edge_type, edge_id, head_json={}, tail_json={}):
    querys = [
        # f'insert $head isa node, has name "{head_name}", has supply_info_json "{head_json}";',
        # f'insert $tail isa node, has name "{tail_name}", has supply_info_json "{tail_json}";',
        f"""
        insert $head isa node, has name "{head_name}", has supply_info_json "{head_json}";
                $tail isa node, has name "{tail_name}", has supply_info_json "{tail_json}";
                $rel (head: $head, tail: $tail) isa edge, has edge_id {edge_id}, has edge_type "{edge_type}";
        """
    ]
    
    _ = f"""
    insert $rel (head: $head_node, tail: $tail_node) isa edge;
        $rel has edge_id "{edge_id}";
        $rel has edge_type "{edge_type}";
    """
    rela = (
        f'match $head_node isa node, has name "{head_name}"; '
        f'      $tail_node isa node, has name "{tail_name}"; '
        f"insert $rel (head: $head_node, tail: $tail_node) isa edge;"
    )
    # querys.append(rela)
    print(querys[0])
    return querys


def insert_triplets(csv_path, host='35.234.48.188', port=48555, batch_query=100):
    ccnet_triplets = []
    with open(csv_path, 'r') as f:
        for line in f:
            _, relation, entity_a, entity_b, meta = line.split('\t')
            entity_a = entity_a.replace('/c/en/', '').split('/')[0]
            entity_b = entity_b.replace('/c/en/', '').split('/')[0]
            meta = meta.replace('\"', '\'')
            ccnet_triplets.append([relation, entity_a, entity_b, meta])
    
    # random.shuffle(ccnet_triplets)
    
    with GraknClient(uri=f"{host}:{port}") as client:
        with client.session(keyspace = "conceptnet") as session:
            transaction = session.transaction().write()
            
            for i, triplet in enumerate(ccnet_triplets):
                logger.info(f"[{i}/{len(ccnet_triplets)}]")
                # with session.transaction().write() as transaction:
                relation, entity_a, entity_b, meta = triplet
                ql_str_list = gql_insert_node(entity_a, entity_b, relation, i, head_json="", tail_json="")
                for ql_str in ql_str_list:
                    transaction.query(ql_str)
                # if i > 10:
                #     break
                if i % batch_query == 0:
                    transaction.commit()
                    transaction = session.transaction().write()

if __name__ == "__main__":
    # define_schema()
    delete_all()
    insert_triplets('/home/ron/Downloads/hateful_meme_cache/conceptnet-en-5.7.0.csv')