// strangemap 의 TS 데이터(SEOUL_PLACES / THEME_COURSES)를 lewisai 의 JSON 으로 export.
// 사용: node scripts/export_data.mjs [STRANGEMAP_DIR]
//   기본 STRANGEMAP_DIR = /Users/seodonghwi/strangemap
//
// TS 객체 리터럴(주석/트레일링 콤마 포함)을 그대로 잘라 eval 한다.
import fs from "node:fs";
import path from "node:path";

const SRC = process.argv[2] || "/Users/seodonghwi/strangemap";
const OUT = path.resolve(path.dirname(new URL(import.meta.url).pathname), "../data/raw");
fs.mkdirSync(OUT, { recursive: true });

function extractArray(filePath, marker) {
  const src = fs.readFileSync(filePath, "utf8");
  const eq = src.indexOf("=", src.indexOf(marker));
  const start = src.indexOf("[", eq);
  let depth = 0, inStr = false, q = "", esc = false;
  for (let i = start; i < src.length; i++) {
    const c = src[i];
    if (inStr) {
      if (esc) esc = false;
      else if (c === "\\") esc = true;
      else if (c === q) inStr = false;
    } else if (c === '"' || c === "'" || c === "`") { inStr = true; q = c; }
    else if (c === "[") depth++;
    else if (c === "]") { depth--; if (depth === 0) {
      const literal = src.slice(start, i + 1);
      // eslint-disable-next-line no-eval
      return eval("(" + literal + ")");
    } }
  }
  throw new Error(`배열을 찾지 못함: ${marker}`);
}

const places = extractArray(path.join(SRC, "src/lib/seoulPlaces.ts"), "SEOUL_PLACES");
fs.writeFileSync(path.join(OUT, "seoul_places.json"), JSON.stringify(places, null, 2), "utf8");
console.log(`seoul_places.json: ${places.length}개`);

try {
  const courses = extractArray(path.join(SRC, "src/data/themeCourses.ts"), "THEME_COURSES");
  fs.writeFileSync(path.join(OUT, "theme_courses.json"), JSON.stringify(courses, null, 2), "utf8");
  console.log(`theme_courses.json: ${courses.length}개`);
} catch (e) {
  console.warn("코스 export 건너뜀:", e.message);
}
