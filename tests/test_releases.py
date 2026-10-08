from bot.releases import best_versions, parse_release, quality_key

KHRUANGBIN = "Khruangbin - Con todo el mundo [2018] [Album] FLAC / 24bit Lossless / WEB / 2018 [Orpheus]"
CD = "2Pac - All Eyez on Me [1996] [Album] FLAC / Lossless / Log (100%) / Cue / CD / 1996 [Orpheus]"
REISSUE = "2Pac - All Eyez on Me [1996] [Album] FLAC / Lossless / WEB / 2005 Reissue [Orpheus]"
DELUXE = "Jacob Collier and Metropole Orkest - Djesse Vol. 4 [2024] [Album] FLAC / 24bit Lossless / WEB / 2024 Deluxe [Orpheus]"
VINYL = "The Murlocs - Young Blindness [2016] [Album] FLAC / 24bit Lossless / Vinyl [Orpheus]"
MIX = "Bonobo - LateNightTales: Bonobo [2013] [DJ Mix] FLAC / Lossless / WEB / 2013 Deluxe Edition [Orpheus]"
BALTHVS = "BALTHVS - Third Vibration [2023] [Album] FLAC / Lossless / WEB [Orpheus]"


def test_parse_fields():
    r = parse_release(KHRUANGBIN)
    assert (r.artist, r.title, r.year, r.kind) == ("Khruangbin", "Con todo el mundo", 2018, "Album")
    assert (r.encoding, r.media, r.edition) == ("24bit Lossless", "WEB", "")
    assert not r.vinyl and not r.special and r.log is None and not r.cue


def test_parse_cd_and_editions():
    cd = parse_release(CD)
    assert cd.log == 100 and cd.cue and cd.media == "CD" and cd.edition == ""
    assert parse_release(REISSUE).edition == "Reissue"
    assert parse_release(DELUXE).edition == "Deluxe"
    assert parse_release(DELUXE).special
    assert parse_release(MIX).edition == "Deluxe Edition"
    assert parse_release(MIX).kind == "DJ Mix"
    assert parse_release(VINYL).vinyl
    assert parse_release(BALTHVS).edition == ""


def test_unparseable():
    assert parse_release("random torrent name FLAC") is None


def _r(media, enc="Lossless", edition=""):
    suffix = f" / {edition}" if edition else ""
    return parse_release(f"A - B [2020] [Album] FLAC / {enc} / {media}{suffix} [X]")


def test_quality_ordering():
    assert quality_key(_r("WEB", "24bit Lossless")) > quality_key(_r("WEB"))
    assert quality_key(_r("WEB")) > quality_key(_r("Vinyl", "24bit Lossless"))
    assert quality_key(_r("Vinyl", "24bit Lossless"), "Vinyl") > quality_key(_r("WEB"), "Vinyl")
    assert quality_key(_r("WEB", edition="2020 Deluxe")) > quality_key(_r("WEB"))


def test_best_versions_groups():
    out = best_versions([CD, REISSUE, "junk", KHRUANGBIN])
    assert [r.artist for r in out] == ["2Pac", "Khruangbin"]
    assert out[0].name == CD
    live = "2Pac - All Eyez on Me [1996] [Live album] FLAC / Lossless / WEB [X]"
    assert len(best_versions([CD, live])) == 2


def test_vinyl_only_group_still_returned():
    out = best_versions([VINYL])
    assert len(out) == 1 and out[0].vinyl


def test_sacd_and_requested_media():
    from bot.releases import parse_release, quality_key, best_versions

    sacd = "Miles Davis - Kind of Blue [1959] [Album] FLAC / 24bit Lossless / SACD / 2010 [Orpheus]"
    web = "Miles Davis - Kind of Blue [1959] [Album] FLAC / 24bit Lossless / WEB / 2015 [Orpheus]"
    cd = "Miles Davis - Kind of Blue [1959] [Album] FLAC / Lossless / Log (100%) / Cue / CD / 1997 [Orpheus]"
    vinyl = "Miles Davis - Kind of Blue [1959] [Album] FLAC / 24bit Lossless / Vinyl [Orpheus]"
    names = [web, cd, sacd, vinyl]
    assert best_versions(names)[0].name == sacd
    assert best_versions(names, media="CD")[0].name == cd
    assert best_versions(names, media="WEB")[0].name == web
    assert best_versions(names, media="Vinyl")[0].name == vinyl
    # A perfect CD rip beats WEB at the same bit depth.
    cd16 = parse_release(cd)
    web16 = parse_release(web.replace("24bit ", ""))
    assert quality_key(cd16) > quality_key(web16)
