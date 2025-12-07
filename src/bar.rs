use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

pub struct MultiProgressBar {
    style: ProgressStyle,
    multi_prog: MultiProgress,
    global_bar: Option<ProgressBar>,
    train_bar: Option<ProgressBar>,
    val_bar: Option<ProgressBar>,
    test_bar: Option<ProgressBar>,
}
impl MultiProgressBar {
    pub fn new(train_count: u64, val_count: u64, test_count: u64) -> Self {
        let style = ProgressStyle::with_template(
            "{prefix} {spinner} {wide_bar} {pos}/{len} ({percent}%) \
             [rate: {per_sec:2} | elapsed: {elapsed} | ETA: {eta}]")
            .unwrap().progress_chars("█▉▊▋▌▍▎▏  ");
        let multi_prog = MultiProgress::new();
        let global_count = train_count.checked_add(val_count).unwrap().checked_add(test_count).unwrap();
        let mut s = Self { style, multi_prog, global_bar: None, train_bar: None, val_bar: None, test_bar: None };
        s.global_bar.replace(s.new_bar_with_count_and_prefix(global_count, "Overall          "));
        s
    }
    pub fn new_bar_with_count_and_prefix(&self, count: u64, prefix: &str) -> ProgressBar {
        let bar = self.multi_prog.add(
            ProgressBar::new(count).with_style(self.style.clone()).with_prefix(prefix.to_string()));
        bar.enable_steady_tick(std::time::Duration::from_millis(50));
        bar
    }
    pub fn start_train_bar(&mut self, count: u64) {
        self.train_bar = Some(self.multi_prog.add(self.new_bar_with_count_and_prefix(count, "Training subset  ")));
    }
    pub fn start_val_bar(&mut self, count: u64) {
        self.val_bar = Some(self.multi_prog.add(self.new_bar_with_count_and_prefix(count, "Validation subset")));
    }
    pub fn start_test_bar(&mut self, count: u64) {
        self.test_bar = Some(self.multi_prog.add(self.new_bar_with_count_and_prefix(count, "Testing subset   ")));
    }
    pub fn finish_train_bar(&self) {
        self.train_bar.as_ref().expect("Did not start bar.").finish();
    }
    pub fn finish_val_bar(&self) {
        self.val_bar.as_ref().expect("Did not start bar.").finish();
    }
    pub fn finish_test_bar(&self) {
        self.test_bar.as_ref().expect("Did not start bar.").finish();
    }
    pub fn finish(&self) {
        self.global_bar.as_ref().expect("Did not start bar.").finish();
    }
    pub fn inc_train_callback(&self) -> impl Fn() -> () {
        || { self.train_bar.as_ref().expect("Did not start bar before callback.").inc(1);
             self.global_bar.as_ref().unwrap().inc(1); }
    }
    pub fn inc_val_callback(&self) -> impl Fn() -> () {
        || { self.val_bar.as_ref().expect("Did not start bar before callback.").inc(1);
             self.global_bar.as_ref().unwrap().inc(1); }
    }
    pub fn inc_test_callback(&self) -> impl Fn() -> () {
        || { self.test_bar.as_ref().expect("Did not start bar before callback.").inc(1);
             self.global_bar.as_ref().unwrap().inc(1); }
    }
}
