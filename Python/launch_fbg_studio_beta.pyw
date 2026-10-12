"""Opt-in Beta launcher; does not start acquisition or install firmware."""

from beta_profile import enable_beta, seed_beta_data

enable_beta()
seed_beta_data()

from fbg_studio_launcher import main


if __name__ == '__main__':
    raise SystemExit(main())
