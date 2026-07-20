import { useState, type FormEvent } from "react";

import { validatePerformanceDateRange } from "../api/performance";

interface DateRangeFilterProps {
  readonly startDate: string;
  readonly endDate: string;
  readonly availableStartDate: string | null;
  readonly availableEndDate: string | null;
  readonly isLoading: boolean;
  readonly onStartDateChange: (value: string) => void;
  readonly onEndDateChange: (value: string) => void;
  readonly onUpdate: () => void;
}

function validateAvailableRange(
  startDate: string,
  endDate: string,
  availableStartDate: string | null,
  availableEndDate: string | null,
): string | null {
  const dateRangeError = validatePerformanceDateRange(startDate, endDate);
  if (dateRangeError) {
    return dateRangeError;
  }

  if (
    startDate &&
    availableStartDate &&
    availableEndDate &&
    (startDate < availableStartDate || startDate > availableEndDate)
  ) {
    return `Start date must be within the available range (${availableStartDate} to ${availableEndDate}).`;
  }

  if (
    endDate &&
    availableStartDate &&
    availableEndDate &&
    (endDate < availableStartDate || endDate > availableEndDate)
  ) {
    return `End date must be within the available range (${availableStartDate} to ${availableEndDate}).`;
  }

  return null;
}

export function DateRangeFilter({
  startDate,
  endDate,
  availableStartDate,
  availableEndDate,
  isLoading,
  onStartDateChange,
  onEndDateChange,
  onUpdate,
}: DateRangeFilterProps) {
  const [hasSubmitted, setHasSubmitted] = useState(false);
  const validationMessage = validateAvailableRange(
    startDate,
    endDate,
    availableStartDate,
    availableEndDate,
  );

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setHasSubmitted(true);
    if (!validationMessage) {
      onUpdate();
    }
  }

  const errorId = "date-range-error";

  return (
    <form
      className="date-range-filter"
      aria-busy={isLoading}
      noValidate
      onSubmit={handleSubmit}
    >
      <div className="date-range-filter__field">
        <label htmlFor="start-date">Start date</label>
        <input
          id="start-date"
          type="date"
          min={availableStartDate ?? undefined}
          max={availableEndDate ?? undefined}
          value={startDate}
          aria-describedby={hasSubmitted && validationMessage ? errorId : undefined}
          aria-invalid={hasSubmitted && validationMessage ? true : undefined}
          onChange={(event) => onStartDateChange(event.target.value)}
        />
      </div>
      <div className="date-range-filter__field">
        <label htmlFor="end-date">End date</label>
        <input
          id="end-date"
          type="date"
          min={availableStartDate ?? undefined}
          max={availableEndDate ?? undefined}
          value={endDate}
          aria-describedby={hasSubmitted && validationMessage ? errorId : undefined}
          aria-invalid={hasSubmitted && validationMessage ? true : undefined}
          onChange={(event) => onEndDateChange(event.target.value)}
        />
      </div>
      <button type="submit" disabled={isLoading}>
        {isLoading ? "Updating…" : "Update dashboard"}
      </button>
      {hasSubmitted && validationMessage && (
        <p id={errorId} className="date-range-filter__error" role="alert">
          {validationMessage}
        </p>
      )}
      {availableStartDate && availableEndDate && (
        <p className="date-range-filter__hint">
          Available from <time dateTime={availableStartDate}>{availableStartDate}</time>{" "}
          to <time dateTime={availableEndDate}>{availableEndDate}</time>. Leave either
          field empty for an open-ended filter.
        </p>
      )}
    </form>
  );
}
