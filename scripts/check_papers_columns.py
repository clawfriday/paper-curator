import db


def main() -> None:
    with db.get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'papers'
                ORDER BY column_name
                """
            )
            cols = [r[0] for r in cur.fetchall()]

    print("papers columns:")
    for c in cols:
        print(f"  {c}")
    print("")
    print("Has abbreviation column:", "abbreviation" in cols)


if __name__ == "__main__":
    main()
