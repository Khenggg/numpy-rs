//! Calendar-aware date/time scalars. Fixed-unit conversion floors negative
//! timestamps like NumPy. NaT is represented by i64::MIN and propagates.
use numpy_types::{NumPyError, Result};
use std::{collections::BTreeSet, fmt, str::FromStr};
fn invalid(s: &str) -> NumPyError {
    NumPyError::InvalidValue(s.into())
}
const NAT: i64 = i64::MIN;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TimeUnit {
    Year,
    Month,
    Week,
    Day,
    Hour,
    Minute,
    Second,
    Millisecond,
    Microsecond,
    Nanosecond,
    Picosecond,
    Femtosecond,
    Attosecond,
}
impl TimeUnit {
    fn attoseconds(self) -> Option<i128> {
        Some(match self {
            Self::Year | Self::Month => return None,
            Self::Week => 604800_000_000_000_000_000_000,
            Self::Day => 86400_000_000_000_000_000_000,
            Self::Hour => 3600_000_000_000_000_000_000,
            Self::Minute => 60_000_000_000_000_000_000,
            Self::Second => 1_000_000_000_000_000_000,
            Self::Millisecond => 1_000_000_000_000_000,
            Self::Microsecond => 1_000_000_000_000,
            Self::Nanosecond => 1_000_000_000,
            Self::Picosecond => 1_000_000,
            Self::Femtosecond => 1_000,
            Self::Attosecond => 1,
        })
    }
}
impl FromStr for TimeUnit {
    type Err = NumPyError;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s {
            "Y" => Self::Year,
            "M" => Self::Month,
            "W" => Self::Week,
            "D" => Self::Day,
            "h" => Self::Hour,
            "m" => Self::Minute,
            "s" => Self::Second,
            "ms" => Self::Millisecond,
            "us" => Self::Microsecond,
            "ns" => Self::Nanosecond,
            "ps" => Self::Picosecond,
            "fs" => Self::Femtosecond,
            "as" => Self::Attosecond,
            _ => return Err(invalid("unknown time unit")),
        })
    }
}
fn checked(value: i128) -> Result<i64> {
    let value = i64::try_from(value).map_err(|_| invalid("date/time overflow"))?;
    if value == NAT {
        return Err(invalid("date/time overflow into NaT"));
    }
    Ok(value)
}
fn days_from_civil(year: i128, month: i128, day: i128) -> i128 {
    let y = year - i128::from(month <= 2);
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let m = month + if month > 2 { -3 } else { 9 };
    let doy = (153 * m + 2) / 5 + day - 1;
    era * 146097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719468
}
fn civil_from_days(days: i128) -> (i128, i128, i128) {
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    (y + i128::from(month <= 2), month, day)
}
fn month_length(year: i128, month: i128) -> i128 {
    match month {
        2 => {
            if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) {
                29
            } else {
                28
            }
        }
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}
#[derive(Clone, Copy, Debug)]
pub struct DateTime64 {
    pub value: i64,
    pub unit: TimeUnit,
}
#[derive(Clone, Copy, Debug)]
pub struct TimeDelta64 {
    pub value: i64,
    pub unit: TimeUnit,
}
fn subday(value: i64, unit: TimeUnit) -> i128 {
    let day = 86_400_000_000_000_000_000_000i128;
    match unit.attoseconds() {
        Some(factor) if factor < day => (value as i128).rem_euclid(day / factor) * factor,
        _ => 0,
    }
}
impl PartialEq for DateTime64 {
    fn eq(&self, other: &Self) -> bool {
        !self.is_nat()
            && !other.is_nat()
            && self.day().ok() == other.day().ok()
            && subday(self.value, self.unit) == subday(other.value, other.unit)
    }
}
impl PartialEq for TimeDelta64 {
    fn eq(&self, other: &Self) -> bool {
        if self.is_nat() || other.is_nat() {
            return false;
        }
        match (self.unit.attoseconds(), other.unit.attoseconds()) {
            (None, None) => {
                let a = self.value as i128 * if self.unit == TimeUnit::Year { 12 } else { 1 };
                let b = other.value as i128 * if other.unit == TimeUnit::Year { 12 } else { 1 };
                a == b
            }
            (Some(_), Some(_)) => {
                DateTime64::new(self.value, self.unit) == DateTime64::new(other.value, other.unit)
            }
            _ => false,
        }
    }
}
impl DateTime64 {
    pub const fn new(value: i64, unit: TimeUnit) -> Self {
        Self { value, unit }
    }
    pub const fn nat(unit: TimeUnit) -> Self {
        Self::new(NAT, unit)
    }
    pub fn is_nat(self) -> bool {
        self.value == NAT
    }
    fn day(self) -> Result<i128> {
        if self.is_nat() {
            return Err(invalid("NaT has no calendar date"));
        }
        Ok(match self.unit {
            TimeUnit::Year => days_from_civil(self.value as i128 + 1970, 1, 1),
            TimeUnit::Month => {
                let m = self.value as i128;
                days_from_civil(1970 + m.div_euclid(12), m.rem_euclid(12) + 1, 1)
            }
            _ => {
                let factor = self
                    .unit
                    .attoseconds()
                    .ok_or_else(|| invalid("invalid time unit"))?;
                // Divide before multiplying where possible to avoid i128 overflow.
                let day = TimeUnit::Day
                    .attoseconds()
                    .ok_or_else(|| invalid("invalid day unit"))?;
                if factor >= day {
                    (self.value as i128) * (factor / day)
                } else {
                    (self.value as i128).div_euclid(day / factor)
                }
            }
        })
    }
    pub fn cast(self, unit: TimeUnit) -> Result<Self> {
        if self.is_nat() {
            return Ok(Self::nat(unit));
        }
        if unit == self.unit {
            return Ok(self);
        }
        let value = match (unit.attoseconds(), self.unit.attoseconds()) {
            (Some(to), Some(from)) => {
                if from >= to {
                    (self.value as i128)
                        .checked_mul(from / to)
                        .ok_or_else(|| invalid("date/time conversion overflow"))?
                } else {
                    (self.value as i128).div_euclid(to / from)
                }
            }
            (None, _) => {
                let (y, m, _) = civil_from_days(self.day()?);
                if unit == TimeUnit::Year {
                    y - 1970
                } else {
                    (y - 1970) * 12 + m - 1
                }
            }
            (Some(to), None) => {
                let day = self.day()?;
                let factor = TimeUnit::Day
                    .attoseconds()
                    .ok_or_else(|| invalid("invalid unit"))?;
                if factor >= to {
                    day.checked_mul(factor / to)
                        .ok_or_else(|| invalid("date/time conversion overflow"))?
                } else {
                    day.div_euclid(to / factor)
                }
            }
        };
        Ok(Self::new(checked(value)?, unit))
    }
    pub fn parse(text: &str, unit: TimeUnit) -> Result<Self> {
        if text == "NaT" {
            return Ok(Self::nat(unit));
        }
        let (date, time) = text
            .split_once('T')
            .or_else(|| text.split_once(' '))
            .map_or((text, None), |(d, t)| (d, Some(t)));
        let (sign, date) = date.strip_prefix('-').map_or((1i128, date), |s| (-1, s));
        let parts: Vec<_> = date.rsplitn(3, '-').collect();
        let (year, month, day) = match parts.as_slice() {
            [y] => (
                y.parse::<i128>().map_err(|_| invalid("invalid year"))?,
                1,
                1,
            ),
            [m, y] => (
                y.parse::<i128>().map_err(|_| invalid("invalid year"))?,
                m.parse::<i128>().map_err(|_| invalid("invalid month"))?,
                1,
            ),
            [d, m, y] => (
                y.parse::<i128>().map_err(|_| invalid("invalid year"))?,
                m.parse::<i128>().map_err(|_| invalid("invalid month"))?,
                d.parse::<i128>().map_err(|_| invalid("invalid day"))?,
            ),
            _ => return Err(invalid("invalid ISO date")),
        };
        let year = year
            .checked_mul(sign)
            .ok_or_else(|| invalid("year overflow"))?;
        if !(-1_000_000_000_000_000_000..=1_000_000_000_000_000_000).contains(&year)
            || !(1..=12).contains(&month)
            || day < 1
            || day > month_length(year, month)
        {
            return Err(invalid("invalid calendar date"));
        }
        let days = days_from_civil(year, month, day);
        let mut result = Self::new(checked(days)?, TimeUnit::Day);
        if let Some(time) = time {
            let time = time.strip_suffix('Z').unwrap_or(time);
            let parts: Vec<_> = time.split(':').collect();
            if parts.is_empty() || parts.len() > 3 {
                return Err(invalid("invalid ISO time"));
            }
            let h = parts[0]
                .parse::<u32>()
                .map_err(|_| invalid("invalid hour"))?;
            let m = parts
                .get(1)
                .copied()
                .unwrap_or("0")
                .parse::<u32>()
                .map_err(|_| invalid("invalid minute"))?;
            let sec = parts.get(2).copied().unwrap_or("0");
            let (s, f) = sec.split_once('.').unwrap_or((sec, ""));
            let s = s.parse::<u32>().map_err(|_| invalid("invalid second"))?;
            if h > 23 || m > 59 || s > 59 || f.len() > 18 || !f.bytes().all(|b| b.is_ascii_digit())
            {
                return Err(invalid("invalid ISO time"));
            }
            let frac = if f.is_empty() {
                0
            } else {
                f.parse::<i128>().map_err(|_| invalid("invalid fraction"))?
                    * 10i128.pow((18 - f.len()) as u32)
            };
            if unit.attoseconds().is_some() {
                let factor = unit
                    .attoseconds()
                    .ok_or_else(|| invalid("invalid time unit"))?;
                let day_as = TimeUnit::Day
                    .attoseconds()
                    .ok_or_else(|| invalid("invalid day unit"))?;
                let second_as = 1_000_000_000_000_000_000i128;
                let day_seconds = (h * 3600 + m * 60 + s) as i128 * second_as + frac;
                let value = if factor <= day_as {
                    days.checked_mul(day_as / factor)
                        .and_then(|v| v.checked_add(day_seconds.div_euclid(factor)))
                        .ok_or_else(|| invalid("date/time overflow"))?
                } else {
                    days.div_euclid(factor / day_as)
                };
                result = Self::new(checked(value)?, unit);
            }
        }
        result.cast(unit)
    }
    pub fn add(self, delta: TimeDelta64) -> Result<Self> {
        let unit = match (self.unit.attoseconds(), delta.unit.attoseconds()) {
            (Some(a), Some(b)) => {
                if a <= b {
                    self.unit
                } else {
                    delta.unit
                }
            }
            (None, Some(_)) => delta.unit,
            (None, None) => {
                if self.unit == TimeUnit::Month || delta.unit == TimeUnit::Month {
                    TimeUnit::Month
                } else {
                    TimeUnit::Year
                }
            }
            (Some(_), None) => {
                return Err(invalid(
                    "cannot add calendar duration to a fixed-unit datetime",
                ));
            }
        };
        if self.is_nat() || delta.is_nat() {
            return Ok(Self::nat(unit));
        }
        let date = self.cast(unit)?;
        let delta = delta.cast(unit)?;
        Ok(Self::new(
            date.value
                .checked_add(delta.value)
                .filter(|v| *v != NAT)
                .ok_or_else(|| invalid("date/time overflow"))?,
            unit,
        ))
    }
    pub fn difference(self, other: Self, unit: TimeUnit) -> Result<TimeDelta64> {
        if self.is_nat() || other.is_nat() {
            return Ok(TimeDelta64::nat(unit));
        }
        if let (Some(a), Some(b), Some(to)) = (
            self.unit.attoseconds(),
            other.unit.attoseconds(),
            unit.attoseconds(),
        ) {
            let common = a.min(b);
            let a = (self.value as i128)
                .checked_mul(a / common)
                .ok_or_else(|| invalid("date difference overflow"))?;
            let b = (other.value as i128)
                .checked_mul(b / common)
                .ok_or_else(|| invalid("date difference overflow"))?;
            let difference = a
                .checked_sub(b)
                .ok_or_else(|| invalid("date difference overflow"))?;
            let result = if common >= to {
                difference
                    .checked_mul(common / to)
                    .ok_or_else(|| invalid("date difference overflow"))?
            } else {
                difference.div_euclid(to / common)
            };
            return Ok(TimeDelta64::new(checked(result)?, unit));
        }
        let common = if self.unit.attoseconds().is_none() && other.unit.attoseconds().is_none() {
            TimeUnit::Month
        } else if let Some(a) = self.unit.attoseconds() {
            let _ = a;
            self.unit
        } else {
            other.unit
        };
        let (a, b) = (self.cast(common)?, other.cast(common)?);
        TimeDelta64::new(checked(a.value as i128 - b.value as i128)?, common).cast(unit)
    }
}
impl TimeDelta64 {
    pub const fn new(value: i64, unit: TimeUnit) -> Self {
        Self { value, unit }
    }
    pub const fn nat(unit: TimeUnit) -> Self {
        Self::new(NAT, unit)
    }
    pub fn is_nat(self) -> bool {
        self.value == NAT
    }
    pub fn cast(self, unit: TimeUnit) -> Result<Self> {
        if self.is_nat() {
            return Ok(Self::nat(unit));
        }
        if self.unit == unit {
            return Ok(self);
        }
        let value = match (self.unit.attoseconds(), unit.attoseconds()) {
            (Some(from), Some(to)) => {
                if from >= to {
                    (self.value as i128)
                        .checked_mul(from / to)
                        .ok_or_else(|| invalid("timedelta overflow"))?
                } else {
                    (self.value as i128).div_euclid(to / from)
                }
            }
            (None, None) => {
                if self.unit == TimeUnit::Year {
                    self.value as i128 * 12
                } else {
                    (self.value as i128).div_euclid(12)
                }
            }
            _ => {
                return Err(invalid(
                    "cannot convert between calendar and fixed duration units",
                ));
            }
        };
        Ok(Self::new(checked(value)?, unit))
    }
    pub fn add(self, other: Self) -> Result<Self> {
        let unit = match (self.unit.attoseconds(), other.unit.attoseconds()) {
            (Some(a), Some(b)) => {
                if a <= b {
                    self.unit
                } else {
                    other.unit
                }
            }
            (None, None) => {
                if self.unit == TimeUnit::Month || other.unit == TimeUnit::Month {
                    TimeUnit::Month
                } else {
                    TimeUnit::Year
                }
            }
            _ => return Err(invalid("cannot add calendar and fixed durations")),
        };
        if self.is_nat() || other.is_nat() {
            return Ok(Self::nat(unit));
        }
        let a = self.cast(unit)?;
        let b = other.cast(unit)?;
        Ok(Self::new(checked(a.value as i128 + b.value as i128)?, unit))
    }
}
impl fmt::Display for DateTime64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_nat() {
            return f.write_str("NaT");
        }
        let days = self.day().map_err(|_| fmt::Error)?;
        let (y, m, d) = civil_from_days(days);
        write!(f, "{y:04}")?;
        if self.unit == TimeUnit::Year {
            return Ok(());
        }
        write!(f, "-{m:02}")?;
        if self.unit == TimeUnit::Month {
            return Ok(());
        }
        write!(f, "-{d:02}")?;
        let factor = self.unit.attoseconds().ok_or(fmt::Error)?;
        let day = TimeUnit::Day.attoseconds().ok_or(fmt::Error)?;
        if factor >= day {
            return Ok(());
        }
        let ticks = (self.value as i128).rem_euclid(day / factor);
        let total = ticks * factor;
        let seconds = total / 1_000_000_000_000_000_000;
        write!(f, "T{:02}", seconds / 3600)?;
        if self.unit == TimeUnit::Hour {
            return Ok(());
        }
        write!(f, ":{:02}", (seconds / 60) % 60)?;
        if self.unit == TimeUnit::Minute {
            return Ok(());
        }
        write!(f, ":{:02}", seconds % 60)?;
        let digits = match self.unit {
            TimeUnit::Millisecond => 3,
            TimeUnit::Microsecond => 6,
            TimeUnit::Nanosecond => 9,
            TimeUnit::Picosecond => 12,
            TimeUnit::Femtosecond => 15,
            TimeUnit::Attosecond => 18,
            _ => 0,
        };
        if digits > 0 {
            let fraction = (total % 1_000_000_000_000_000_000) / 10i128.pow(18 - digits);
            write!(f, ".{fraction:0width$}", width = digits as usize)?;
        }
        Ok(())
    }
}
#[derive(Clone, Debug)]
pub struct BusinessCalendar {
    weekmask: [bool; 7],
    holidays: BTreeSet<i64>,
}
#[derive(Clone, Copy, Debug)]
pub enum Roll {
    Raise,
    NaT,
    Forward,
    Backward,
    ModifiedFollowing,
    ModifiedPreceding,
}
impl Default for BusinessCalendar {
    fn default() -> Self {
        Self {
            weekmask: [true, true, true, true, true, false, false],
            holidays: BTreeSet::new(),
        }
    }
}
fn busday_unit_ok(unit: TimeUnit) -> bool {
    matches!(
        unit,
        TimeUnit::Year | TimeUnit::Month | TimeUnit::Week | TimeUnit::Day
    )
}
fn busday_date(date: DateTime64) -> Result<i64> {
    if !busday_unit_ok(date.unit) {
        return Err(invalid("cannot safely cast sub-day datetime unit to days"));
    }
    Ok(date.cast(TimeUnit::Day)?.value)
}
impl BusinessCalendar {
    pub fn new(weekmask: [bool; 7], holidays: &[DateTime64]) -> Result<Self> {
        if !weekmask.iter().any(|&v| v) {
            return Err(invalid("weekmask must enable at least one day"));
        }
        let mut parsed = BTreeSet::new();
        for d in holidays {
            if !busday_unit_ok(d.unit) {
                return Err(invalid("cannot safely cast sub-day datetime unit to days"));
            }
            if !d.is_nat() {
                parsed.insert(d.cast(TimeUnit::Day)?.value);
            }
        }
        Ok(Self {
            weekmask,
            holidays: parsed,
        })
    }
    fn valid_day(&self, day: i64) -> bool {
        self.weekmask[((day as i128 + 3).rem_euclid(7)) as usize] && !self.holidays.contains(&day)
    }
    fn count_forward(&self, a: i128, b: i128) -> i128 {
        if a >= b {
            return 0;
        }
        let span = b - a;
        let weeks = span / 7;
        let per = self.weekmask.iter().filter(|&&v| v).count() as i128;
        let mut count = weeks * per;
        for k in 0..(span % 7) {
            let day = a + weeks * 7 + k;
            if self.weekmask[((day + 3).rem_euclid(7)) as usize] {
                count += 1;
            }
        }
        let lo = i64::try_from(a).unwrap_or(i64::MIN);
        if b <= i64::MAX as i128 {
            let hi = i64::try_from(b).unwrap_or(i64::MAX);
            if lo < hi {
                for &h in self.holidays.range(lo..hi) {
                    if self.weekmask[((h as i128 + 3).rem_euclid(7)) as usize] {
                        count -= 1;
                    }
                }
            }
        } else {
            for &h in self.holidays.range(lo..=i64::MAX) {
                if (h as i128) < b && self.weekmask[((h as i128 + 3).rem_euclid(7)) as usize] {
                    count -= 1;
                }
            }
        }
        count
    }
    pub fn is_busday(&self, date: DateTime64) -> Result<bool> {
        if date.is_nat() {
            return Ok(false);
        }
        Ok(self.valid_day(date.cast(TimeUnit::Day)?.value))
    }
    pub fn busday_count(&self, begin: DateTime64, end: DateTime64) -> Result<i64> {
        if !busday_unit_ok(begin.unit) || !busday_unit_ok(end.unit) {
            return Err(invalid("cannot safely cast sub-day datetime unit to days"));
        }
        if begin.is_nat() || end.is_nat() {
            return Err(invalid("business-day count cannot use NaT"));
        }
        let (a, b) = (busday_date(begin)?, busday_date(end)?);
        if a > b {
            return self
                .busday_count(end, begin)
                .map(|n| -n + i64::from(self.valid_day(b)) - i64::from(self.valid_day(a)));
        }
        checked(self.count_forward(a as i128, b as i128))
    }
    pub fn busday_offset(&self, date: DateTime64, offset: i64, roll: Roll) -> Result<DateTime64> {
        if !busday_unit_ok(date.unit) {
            return Err(invalid("cannot safely cast sub-day datetime unit to days"));
        }
        if date.is_nat() {
            return match roll {
                Roll::NaT => Ok(DateTime64::nat(TimeUnit::Day)),
                _ => Err(invalid("business-day offset cannot use NaT")),
            };
        }
        let mut day = busday_date(date)?;
        if !self.valid_day(day) {
            let original = day;
            let direction = match roll {
                Roll::Raise => return Err(invalid("date is not a business day")),
                Roll::NaT => return Ok(DateTime64::nat(TimeUnit::Day)),
                Roll::Forward | Roll::ModifiedFollowing => 1,
                Roll::Backward | Roll::ModifiedPreceding => -1,
            };
            while !self.valid_day(day) {
                day = checked(day as i128 + direction)?;
            }
            if matches!(roll, Roll::ModifiedFollowing | Roll::ModifiedPreceding)
                && civil_from_days(day as i128).1 != civil_from_days(original as i128).1
            {
                day = original;
                while !self.valid_day(day) {
                    day = checked(day as i128 - direction)?;
                }
            }
        }
        if offset == 0 {
            return Ok(DateTime64::new(day, TimeUnit::Day));
        }
        // Binary-search a count boundary, so huge offsets avoid day-by-day work.
        let direction = if offset > 0 { 1i128 } else { -1 };
        let target = offset.unsigned_abs() as i128;
        let max_dist = if direction > 0 {
            (i64::MAX as i128) - (day as i128)
        } else {
            (day as i128) - ((NAT + 1) as i128)
        };
        let raw_hi = target.saturating_mul(7).saturating_add(
            (self.holidays.len() as i128)
                .saturating_mul(7)
                .saturating_add(7),
        );
        let mut lo = 0i128;
        let mut hi = raw_hi.min(max_dist);
        let count_at = |dist: i128| -> i128 {
            let candidate = day as i128 + direction * dist;
            if direction > 0 {
                self.count_forward(day as i128 + 1, candidate + 1)
            } else {
                self.count_forward(candidate, day as i128)
            }
        };
        if hi == max_dist && count_at(hi) < target {
            return Err(invalid("date/time overflow"));
        }
        while lo < hi {
            let mid = (lo + hi) / 2;
            if count_at(mid) < target {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        Ok(DateTime64::new(
            checked(day as i128 + direction * lo)?,
            TimeUnit::Day,
        ))
    }
}
pub fn is_busday(date: DateTime64) -> Result<bool> {
    BusinessCalendar::default().is_busday(date)
}
pub fn busday_count(begin: DateTime64, end: DateTime64) -> Result<i64> {
    BusinessCalendar::default().busday_count(begin, end)
}
pub fn busday_offset(date: DateTime64, offset: i64, roll: Roll) -> Result<DateTime64> {
    BusinessCalendar::default().busday_offset(date, offset, roll)
}
