"""Helpers for validating optimizer evaluation results."""


def ensure_population_has_valid_results(workchain, results, raw_results=None, context="current population"):
    """Abort optimization when every evaluation in a batch returned the penalty."""

    penalty = workchain.extractor.get_penalty()
    if all(result == penalty for result in results):
        if raw_results is not None:
            failed_pks = [item.get("pk") for item in raw_results]
            workchain.report(
                f"All evaluations in {context} returned penalty {penalty}. Submitted process PKs: {failed_pks}"
            )
        else:
            workchain.report(f"All evaluations in {context} returned penalty {penalty}.")
        return workchain.exit_codes.ERROR_NO_VALID_SOLUTION
    return None
