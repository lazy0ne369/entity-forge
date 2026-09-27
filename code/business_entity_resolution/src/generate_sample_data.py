"""
generate_sample_data.py
Creates synthetic train and test TSV files strictly following the Amazon ML Challenge 2026 schema.
Used for pipeline verification, local unit testing, and benchmarking before running on the full dataset.
"""

import os
import random

def create_sample_datasets(base_dir: str = "d:/AWS_ML_Hackathon/data/sample_dataset"):
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    random.seed(42)

    # Base templates for business entities
    us_names = [
        "Apex Global Logistics", "Summit Health Systems", "Blue Horizon Technologies",
        "Pinnacle Financial Services", "Vanguard Retail Partners", "Starlight Media Group",
        "Beacon Industrial Supply", "Atlas Cloud Solutions", "Ironclad Construction",
        "Pacific Coast Marketing", "NextGen Robotics", "Omni Health Diagnostics"
    ]
    us_addresses = [
        "1042 Market St, Suite 400, San Francisco, CA 94103",
        "780 Broadway Ave, Floor 12, New York, NY 10003",
        "2400 Peachtree Rd NW, Atlanta, GA 30309",
        "500 Michigan Ave, Chicago, IL 60611",
        "1200 Westheimer Rd, Houston, TX 77006",
        "8500 Wilshire Blvd, Beverly Hills, CA 90211"
    ]

    in_names = [
        "Tata Consultancy Services", "Reliance Retail Enterprises", "Infosys Digital Solutions",
        "Mahindra Aerospace Ltd", "Adani Logistics Solutions", "Bharti Global Communications",
        "HDFC Financial Corporation", "Wipro Infotech Systems", "Larsen and Toubro Engineering",
        "Bajaj Auto Components", "Sun Pharma Laboratories", "Godrej Consumer Products"
    ]
    in_addresses = [
        "Plot 42, Electronic City Phase 1, Hosur Road, Bengaluru, Karnataka 560100",
        "Bandra Kurla Complex, G Block, Bandra East, Mumbai, Maharashtra 400051",
        "DLF Cyber City, Tower B, Sector 24, Gurugram, Haryana 122002",
        "Hi-Tech City, Madhapur, Near Cyber Towers, Hyderabad, Telangana 500081",
        "Salt Lake Sector V, Block EP & GP, Bidhannagar, Kolkata, West Bengal 700091",
        "Anna Salai, Mount Road, Near Spencer Plaza, Chennai, Tamil Nadu 600002"
    ]

    fr_names = [
        "Société Générale Logistique", "L'Oréal Beauté Solutions", "Carrefour Distribution",
        "Schneider Électrique Systèmes", "TotalEnergies Solutions Nouvelles", "Sanofi Santé France",
        "Danone Produits Frais", "Capgemini Technologies Paris"
    ]
    fr_addresses = [
        "15 Boulevard Haussmann, 75009 Paris, France",
        "28 Rue de la République, 69002 Lyon, France",
        "10 Place de la Bourse, 33000 Bordeaux, France",
        "45 Avenue du Prado, 13006 Marseille, France"
    ]

    def add_noise_to_name(name: str) -> str:
        variations = [
            lambda s: s.replace("Logistics", "Logistics Pvt Ltd"),
            lambda s: s.replace("Technologies", "Tech Inc."),
            lambda s: s.replace("Solutions", "Solns Corp"),
            lambda s: s.replace("Services", "Svcs LLC"),
            lambda s: s.replace("Enterprises", "Ent."),
            lambda s: s.replace("Engineering", "Engg Ltd"),
            lambda s: s.lower(),
            lambda s: s.upper(),
            lambda s: s + " & Co",
            lambda s: s.replace("and", "&")
        ]
        var_func = random.choice(variations)
        res = var_func(name)
        return res

    def add_noise_to_addr(addr: str) -> str:
        variations = [
            lambda s: s.replace("Street", "St.").replace("St,", "St.,"),
            lambda s: s.replace("Road", "Rd.").replace("Avenue", "Ave"),
            lambda s: s.replace("Near ", "Opposite to "),
            lambda s: s.split(",")[0] + ", " + s.split(",")[-1], # dropped middle components
            lambda s: s.replace("Floor 12", "12th Flr"),
            lambda s: s.replace("Suite 400", "Ste 400")
        ]
        return random.choice(variations)(addr)

    # Generate Train Split (US and India)
    train_s1 = []
    train_s2 = []
    train_s3 = []
    train_gt = []

    s1_counter = 1
    s2_counter = 1
    s3_counter = 1

    train_pool = [("US", us_names, us_addresses), ("India", in_names, in_addresses)]
    for country, names, addrs in train_pool:
        for idx in range(len(names)):
            base_name = names[idx]
            base_addr = addrs[idx % len(addrs)]
            s1_id = f"S1-{s1_counter:05d}"
            s1_counter += 1
            train_s1.append((s1_id, base_name, base_addr, country))

            # Match type: 0 = singleton, 1 = match S2, 2 = match S3, 3 = match both S2 and S3
            match_type = random.choice([0, 1, 2, 3])
            matched_ids = []

            if match_type in (1, 3):
                s2_id = f"S2-{s2_counter:05d}"
                s2_counter += 1
                train_s2.append((s2_id, add_noise_to_name(base_name), add_noise_to_addr(base_addr), country))
                matched_ids.append(s2_id)

            if match_type in (2, 3):
                s3_id = f"S3-{s3_counter:05d}"
                s3_counter += 1
                train_s3.append((s3_id, add_noise_to_name(base_name), add_noise_to_addr(base_addr), country))
                matched_ids.append(s3_id)

            train_gt.append((s1_id, ",".join(matched_ids)))

    # Add some random unmatched noise records into S2 and S3
    for _ in range(10):
        country = random.choice(["US", "India"])
        names = us_names if country == "US" else in_names
        addrs = us_addresses if country == "US" else in_addresses
        s2_id = f"S2-{s2_counter:05d}"
        s2_counter += 1
        train_s2.append((s2_id, "Independent Unmatched " + random.choice(names), random.choice(addrs), country))

        s3_id = f"S3-{s3_counter:05d}"
        s3_counter += 1
        train_s3.append((s3_id, "Distractor Store " + random.choice(names), random.choice(addrs), country))

    # Generate Test Split (US, India, and France)
    test_s1 = []
    test_s2 = []
    test_s3 = []
    test_gt_hidden = [] # for internal scoring verification

    s1_t_counter = 1
    s2_t_counter = 1
    s3_t_counter = 1

    test_pool = [("US", us_names, us_addresses), ("India", in_names, in_addresses), ("France", fr_names, fr_addresses)]
    for country, names, addrs in test_pool:
        for idx in range(len(names)):
            base_name = names[idx]
            base_addr = addrs[idx % len(addrs)]
            s1_id = f"S1-{s1_t_counter:05d}"
            s1_t_counter += 1
            test_s1.append((s1_id, base_name, base_addr, country))

            match_type = random.choice([0, 1, 2, 3])
            matched_ids = []

            if match_type in (1, 3):
                s2_id = f"S2-{s2_t_counter:05d}"
                s2_t_counter += 1
                test_s2.append((s2_id, add_noise_to_name(base_name), add_noise_to_addr(base_addr), country))
                matched_ids.append(s2_id)

            if match_type in (2, 3):
                s3_id = f"S3-{s3_t_counter:05d}"
                s3_t_counter += 1
                test_s3.append((s3_id, add_noise_to_name(base_name), add_noise_to_addr(base_addr), country))
                matched_ids.append(s3_id)

            test_gt_hidden.append((s1_id, ",".join(matched_ids)))

    # Add distractors to test
    for _ in range(15):
        country = random.choice(["US", "India", "France"])
        if country == "US":
            names, addrs = us_names, us_addresses
        elif country == "India":
            names, addrs = in_names, in_addresses
        else:
            names, addrs = fr_names, fr_addresses
        s2_id = f"S2-{s2_t_counter:05d}"
        s2_t_counter += 1
        test_s2.append((s2_id, "Global Vendor " + random.choice(names), random.choice(addrs), country))

        s3_id = f"S3-{s3_t_counter:05d}"
        s3_t_counter += 1
        test_s3.append((s3_id, "Marketplace Seller " + random.choice(names), random.choice(addrs), country))

    def write_tsv(filepath: str, rows, header: str):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(header + "\n")
            for row in rows:
                f.write("\t".join(str(c) for c in row) + "\n")

    # Write train files
    write_tsv(os.path.join(train_dir, "train_source1.tsv"), train_s1, "entity_id\tbusiness_name\tbusiness_address\tcountry")
    write_tsv(os.path.join(train_dir, "train_source2.tsv"), train_s2, "entity_id\tbusiness_name\tbusiness_address\tcountry")
    write_tsv(os.path.join(train_dir, "train_source3.tsv"), train_s3, "entity_id\tbusiness_name\tbusiness_address\tcountry")
    write_tsv(os.path.join(train_dir, "train_ground_truth.tsv"), train_gt, "source1_entity_id\tmatched_entity_ids")

    # Write test files
    write_tsv(os.path.join(test_dir, "test_source1.tsv"), test_s1, "entity_id\tbusiness_name\tbusiness_address\tcountry")
    write_tsv(os.path.join(test_dir, "test_source2.tsv"), test_s2, "entity_id\tbusiness_name\tbusiness_address\tcountry")
    write_tsv(os.path.join(test_dir, "test_source3.tsv"), test_s3, "entity_id\tbusiness_name\tbusiness_address\tcountry")
    write_tsv(os.path.join(test_dir, "hidden_test_ground_truth.tsv"), test_gt_hidden, "source1_entity_id\tmatched_entity_ids")

    print(f"Sample dataset successfully generated in: {base_dir}")
    print(f"  Train: S1={len(train_s1)}, S2={len(train_s2)}, S3={len(train_s3)}, GT={len(train_gt)}")
    print(f"  Test:  S1={len(test_s1)}, S2={len(test_s2)}, S3={len(test_s3)}")

if __name__ == "__main__":
    create_sample_datasets()
